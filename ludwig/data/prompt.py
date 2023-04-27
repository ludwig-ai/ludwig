import json
import logging
import os
import string
import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ludwig.backend.base import Backend

from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.models.retrieval import get_retrieval_model, RetrievalModel
from ludwig.utils.fs_utils import get_default_cache_location
from ludwig.utils.types import Series

logger = logging.getLogger(__name__)

DEFAULT_ZERO_SHOT_PROMPT_TEMPLATE = """SAMPLE INPUT: {sample_input}

USER: Complete the following task: {task}

ASSISTANT:
"""


DEFAULT_FEW_SHOT_PROMPT_TEMPLATE = """Below is relevant context:

CONTEXT: {context}

The context is comprised of labeled samples whose embeddings were similar to that of the sample input. The labels in
these samples could aid you in your final prediction. Given this context and no prior knowledge, follow the instructions
below.

SAMPLE INPUT: {sample_input}

USER: Complete the following task: {task}

ASSISTANT:
"""


def index_column(
    retrieval_config: Dict[str, Any],
    col_name: str,
    dataset_cols: Dict[str, Series],
    backend: "Backend",
    split_col: Optional[Series] = None,
) -> Tuple[RetrievalModel, str]:
    """Indexes a column for sample retrieval via embedding index lookup.

    This function indexes a column and saves the index artifact to disk. If an index name is provided as part of the
    `retrieval_config`, then the index in the ludwig cache with the corresponding name will be loaded instead of being
    built from scratch.

    To prevent data leakage, a split column must be provided. This ensures that the retrieval model only ever fetches
    samples from the training set.

    To ensure that the index is usable even if the original DataFrame is not available, the columns used to build the
    index are stored as part of the index.

    All operations in this function are performed on pandas objects, which means that you may run out of memory if your
    dataset is large.

    Args:
        retrieval_config (Dict[str, Any]): The retrieval config from the config object.
        col_name (str): The name of the column to index.
        dataset_cols (Dict[str, Series]): A dictionary mapping column names to their corresponding Series. `col_name`
            must be a key in this dictionary. These columns are stored as part of the index to ensure that the index
            is usable even if the original DataFrame is not available.
        df_engine (DataFrameEngine): The engine used to compute the columns into pandas objects.
        split_col (Optional[Series]): A column that indicates whether a sample is part of the training set. A sample
            is in the training set if the value in this column is 0.
    Returns:
        Tuple[RetrievalModel, str]: A tuple containing the retrieval model and the name of the index.
    """
    retrieval_model = get_retrieval_model(
        retrieval_config["type"],
        model_name=retrieval_config["model_name"],
    )

    index_name = retrieval_config["index_name"]
    index_cache_directory = os.path.join(get_default_cache_location(), "index")
    if not os.path.exists(index_cache_directory):
        os.makedirs(index_cache_directory, exist_ok=True)

    if index_name is None:
        if split_col is None:
            raise ValueError("split column must be provided if using retrieval")
        split_col = backend.df_engine.compute(split_col).astype(int)

        # TODO(geoffrey): add support for Dask DataFrames
        df = pd.DataFrame({name: backend.df_engine.compute(col) for name, col in dataset_cols.items()})
        df = df[split_col == 0]  # Ensures that the index is only built on the training set
        retrieval_model.create_dataset_index(df, backend, columns_to_index=[col_name])
        index_name = f"embedding_index_{uuid.uuid4()}"
        logger.info(f"Saving index to cache directory '{index_cache_directory}' with name '{index_name}'")
        retrieval_model.save_index(index_name, cache_directory=index_cache_directory)
    else:
        logger.info(f"Loading index from cache directory '{index_cache_directory}' with name '{index_name}'")
        retrieval_model.load_index(index_name, cache_directory=index_cache_directory)
    return retrieval_model, index_name


def format_input_with_prompt(
    input_col_name: str,
    input_col: Series,
    backend: "Backend",
    task_str: str,
    retrieval_model: Optional[RetrievalModel] = None,
    template: Optional[str] = None,
) -> Series:
    """Returns a new Series with the input column data formatted with the prompt.

    A prompt can either be zero-shot or few-shot. A zero-shot prompt is comprised of some (unlabeled) input and a task
    to be completed given the input. A few-shot prompt additionally includes some dynamically retrieved context, which
    is retrieved using the `retrieval_model.search` function.

    A template can be provided to customize the prompt. The template must be a string with the following fields:
        - sample_input: The input sample.
        - task: The task to be completed given the input.
        - context: The context retrieved by the `search_fn` function. Only required if `search_fn` is provided.
    """
    # determine if this is a few-shot or zero-shot prompt
    # few-shot prompts require a search function that returns samples from some dataset
    is_few_shot = retrieval_model is not None

    # if no template is provided, use the default template
    if template is None:
        if is_few_shot:
            template = DEFAULT_FEW_SHOT_PROMPT_TEMPLATE
        else:
            template = DEFAULT_ZERO_SHOT_PROMPT_TEMPLATE

    # ensure that the prompt template has all required fields
    try:
        _validate_prompt_template(template, is_few_shot=is_few_shot)
    except ValueError as e:
        raise ValueError(f"template invalid for {'few-shot' if is_few_shot else 'zero-shot'} prompt: {e}")

    def generate_prompt(series: pd.Series):
        df = series.to_frame(name=input_col_name)

        if is_few_shot:
            df["context"] = retrieval_model.search(df, backend, k=3, return_data=True)

        df["sample_input"] = df[input_col_name].map(lambda entry: json.dumps(entry, indent=2))
        df["task"] = task_str

        def generate_prompt_for_row(row):
            format_kwargs = {
                "sample_input": row["sample_input"],
                "task": row["task"],
            }
            if is_few_shot:
                format_kwargs["context"] = row["context"]
            return template.format(**format_kwargs)

        df[input_col_name] = df.apply(generate_prompt_for_row, axis=1)
        return df[input_col_name]

    result = backend.df_engine.map_partitions(input_col, generate_prompt, meta=input_col)
    result = backend.df_engine.persist(result)  # persist to prevent re-computation
    return result


def _validate_prompt_template(template: str, is_few_shot: bool):
    """Validates that the template contains the necessary fields for the prompt."""
    if is_few_shot:
        required_fields = {"context", "sample_input", "task"}
    else:
        required_fields = {"sample_input", "task"}
    template_fields = {field for _, field, _, _ in string.Formatter().parse(template) if field is not None}
    missing_fields = required_fields - template_fields
    if missing_fields:
        raise ValueError(f"template is missing the following formattable fields: {missing_fields}")
