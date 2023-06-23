import json
import logging
import os
import string
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ludwig.backend.base import Backend

from ludwig.models.retrieval import df_checksum, get_retrieval_model, RetrievalModel
from ludwig.utils.fs_utils import get_default_cache_location, makedirs, path_exists
from ludwig.utils.types import DataFrame, Series

logger = logging.getLogger(__name__)

CONTEXT = "__context__"
SAMPLE = "__sample__"
TASK = "__task__"

DEFAULT_ZERO_SHOT_PROMPT_TEMPLATE = """SAMPLE INPUT: {__sample__}

USER: Complete the following task: {__task__}

ASSISTANT:
"""


DEFAULT_FEW_SHOT_PROMPT_TEMPLATE = """Below is relevant context:

CONTEXT: {__context__}

CONTEXT is comprised of labeled samples whose embeddings were similar to that of the sample input. The labels in
these samples could aid you in your final prediction. Given this and no prior knowledge, follow the instructions
below.

SAMPLE INPUT: {__sample__}

USER: Complete the following task: {__task__}

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
    if not path_exists(index_cache_directory):
        makedirs(index_cache_directory, exist_ok=True)

    if index_name is None:
        if split_col is None:
            raise ValueError("split column must be provided if using retrieval")
        split_col = backend.df_engine.compute(split_col).astype(int)

        # TODO(geoffrey): add support for Dask DataFrames
        df = pd.DataFrame({name: backend.df_engine.compute(col) for name, col in dataset_cols.items()})
        df = df[split_col == 0]  # Ensures that the index is only built on the training set

        # Even if index name is not provided, we still want to check if an index for this df already exists in cache
        # If it does, load it and return immediately
        index_hash = df_checksum(df)
        index_name = f"embedding_index_{index_hash}"
        if path_exists(os.path.join(index_cache_directory, index_name)):
            logger.info(
                f"Index for this DataFrame with name '{index_name}' already exists. "
                f"Loading index from '{index_cache_directory}'"
            )
            retrieval_model.load_index(index_name, cache_directory=index_cache_directory)
            return retrieval_model, index_name

        # Build index if index name is not provided and index for this df does not already exist in cache
        retrieval_model.create_dataset_index(df, backend, columns_to_index=[col_name])
        logger.info(f"Saving index to cache directory '{index_cache_directory}' with name '{index_name}'")
        retrieval_model.save_index(index_name, cache_directory=index_cache_directory)
    else:
        logger.info(f"Loading index from cache directory '{index_cache_directory}' with name '{index_name}'")
        retrieval_model.load_index(index_name, cache_directory=index_cache_directory)
    return retrieval_model, index_name


def format_input_with_prompt(
    input_col_name: str,
    dataset_df: DataFrame,
    backend: "Backend",
    task_str: str,
    retrieval_model: Optional[RetrievalModel] = None,
    k: int = -1,
    template: Optional[str] = None,
) -> Series:
    """Returns a new Series with the input column data formatted with the prompt.

    A prompt can either be zero-shot or few-shot. A zero-shot prompt is comprised of some (unlabeled) input and a task
    to be completed given the input. A few-shot prompt additionally includes some dynamically retrieved context, which
    is retrieved using the `retrieval_model.search` function.

    A template can be provided to customize the prompt. The template must be a string with the following fields:
        - __sample__ or at least one column from the input dataset: The input sample.
        - __context__: The context retrieved by the `search_fn` function. Only required if `search_fn` is provided.
        - __task__: The task to be completed given the input. Only required if `task` is set in the prompt config.

    Zero-shot example:

    Before formatting:

        input_col = ["I am happy"]
        task_str = "sentiment analysis"

    After formatting:

        input_col = ["SAMPLE INPUT: I am happy\n\nUSER: Complete the following task: sentiment analysis\n\nASSISTANT:"]

    Args:
        input_col_name (str): The name of the input column.
        dataset_df (DataFrame): The input dataset.
        backend (Backend): The backend used for map operations.
        task_str (str): The task to be completed given the input.
        retrieval_model (Optional[RetrievalModel]): The retrieval model used to retrieve context. If provided, the
            prompt will be few-shot. If not provided, the prompt will be zero-shot.
        k (int): The number of samples to retrieve. Only required if `retrieval_model` is provided.
        template (Optional[str]): The template to use for the prompt. If not provided, the default will be used.

    Returns:
        Series: A new Series with the input column data formatted with the prompt.
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
    template_fields, field_to_dtype = _get_template_fields(template)
    try:
        _validate_prompt_template(template_fields, task_str, is_few_shot, dataset_df.columns, input_col_name)
    except ValueError as e:
        raise ValueError(f"template invalid for {'few-shot' if is_few_shot else 'zero-shot'} prompt: {e}")

    def generate_prompt(df: pd.DataFrame):
        if CONTEXT in template_fields:
            df[CONTEXT] = retrieval_model.search(df, backend, k=k, return_data=True)
        if SAMPLE in template_fields:
            # During preprocessing, we're inserting quotes that change the token IDs completely if we
            # don't remove the " from the string. For parity with expected user output, we need to get rid of them.
            # TODO(Arnav): see if there's a way to only remove them if the entry does't have quotes. This currently
            # removes all " from the string (even those not added by json.dumps), which is not ideal.
            df[SAMPLE] = df[input_col_name].map(lambda entry: json.dumps(entry, indent=2).strip('"'))
        if TASK in template_fields:
            df[TASK] = task_str

        def generate_prompt_for_row(row):
            kwargs = {col: field_to_dtype[col](row[col]) for col in template_fields}
            return template.format(**kwargs)

        return df.apply(generate_prompt_for_row, axis=1)

    result = backend.df_engine.map_partitions(dataset_df, generate_prompt, meta=(input_col_name, "object"))
    result = backend.df_engine.persist(result)  # persist to prevent re-computation
    return result


def _validate_prompt_template(
    template_fields: Set[str], task: Optional[str], is_few_shot: bool, columns: List[str], input_col_name: str
):
    """Validates that the template contains the necessary fields for the prompt."""
    if is_few_shot and CONTEXT not in template_fields:
        raise ValueError(f"Prompt template must contain the '{CONTEXT}' field for few-shot learning")

    if task is not None and TASK not in template_fields:
        raise ValueError(f"Prompt template must contain the '{TASK}' field if a task is provided")

    if SAMPLE in template_fields:
        if input_col_name not in columns:
            raise ValueError(
                f"Prompt template contains the '{SAMPLE}' field, "
                f"but the input column '{input_col_name}' is not in the dataset"
            )
    elif not any(col in template_fields for col in columns):
        raise ValueError(
            f"Prompt template must contain either the '{SAMPLE}' field or one of the columns from the dataset"
        )


def _get_template_fields(template: str) -> Tuple[Set[str], Dict[str, Type]]:
    """Returns the fields in the template."""
    parsed = [t for t in string.Formatter().parse(template) if t[1] is not None]
    field_set = {field for _, field, _, _ in parsed}
    dtype_map = {field: _get_dtype(format_spec) for _, field, format_spec, _ in parsed}
    return field_set, dtype_map


def _get_dtype(format_spec: str) -> Type:
    # We need to prepare data in the row for different formatting options.
    # If you have a number like 0.1234 in the DF and you want to format it like {number:.2f} it will fail if the
    # number is represented as a string in the DF. So we need to cast it to a float before formatting.
    if not format_spec:
        return str
    if "f" in format_spec:
        return float
    raise ValueError(f"Unsupported template format spec: {format_spec}")
