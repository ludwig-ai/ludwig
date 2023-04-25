import json
import string
import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.models.retrieval import get_retrieval_model, RetrievalModel
from ludwig.utils.fs_utils import get_default_cache_location
from ludwig.utils.types import Series

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
    df_engine: DataFrameEngine,
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
    index_cache_directory = get_default_cache_location()
    if index_name is None:
        if split_col is None:
            raise ValueError("split column must be provided if using retrieval")
        split_col = df_engine.compute(split_col).astype(int)

        # TODO(geoffrey): add support for Dask DataFrames
        df = pd.DataFrame({name: df_engine.compute(col) for name, col in dataset_cols.items()})
        df = df[split_col == 0]  # Ensures that the index is only built on the training set
        retrieval_model.create_dataset_index(df, columns_to_index=[col_name])
        index_name = f"embedding_index_{uuid.uuid4()}"
        retrieval_model.save_index(index_name, cache_directory=index_cache_directory)
    else:
        retrieval_model.load_index(index_name, cache_directory=index_cache_directory)
    return retrieval_model, index_name


def format_input_with_prompt(
    input_col_name: str,
    input_col: Series,
    task_str: str,
    df_engine: DataFrameEngine,
    search_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    template: Optional[str] = None,
) -> Series:
    """Returns a new Series with the input column data formatted with the prompt.

    A prompt can either be zero-shot or few-shot. A zero-shot prompt is comprised of some (unlabeled) input and a task
    to be completed given the input. A few-shot prompt additionally includes some dynamically retrieved context, which
    is retrieved using the `search_fn` function.

    A template can be provided to customize the prompt. The template must be a string with the following fields:
        - sample_input: The input sample.
        - task: The task to be completed given the input.
        - context: The context retrieved by the `search_fn` function. Only required if `search_fn` is provided.
    
    Args:
        input_col_name (str): The name of the input column.
        input_col (Series): The input column.
        task_str (str): The task to be completed given the input.
        df_engine (DataFrameEngine): The engine used to compute the input column into a pandas object.
        search_fn (Optional[Callable[[str], List[Dict[str, Any]]]]): A function that takes in a sample input and
            returns a list of samples from some dataset. Only required if the prompt is few-shot.
        template (Optional[str]): A template for the prompt. If not provided, the default template will be used.
    Returns:
        Series: A new Series with the input column data formatted with the prompt.
    """
    # determine if this is a few-shot or zero-shot prompt
    # few-shot prompts require a search function that returns samples from some dataset
    is_few_shot = search_fn is not None

    # function for retrieving the context for a given sample.
    # If `search_fn` is not provided, context is omitted.
    if is_few_shot:
        context_fn = partial(get_context, search_fn=search_fn)
    else:
        context_fn = lambda _: ""

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

    # function for getting the sample input. This ensures that only the input_features of a sample are returned
    sample_input_fn = partial(
        get_sample_input,
        input_col_name=input_col_name,
    )

    # function for getting the task.
    task_fn = partial(get_task, task_str=task_str)

    # function for generating the prompt (context + sample input + task) for a given sample
    prompt_fn = partial(
        generate_prompt,
        template=template,
        context_fn=context_fn,
        sample_input_fn=sample_input_fn,
        task_fn=task_fn,
    )

    return df_engine.map_objects(input_col, prompt_fn)


def generate_prompt(
    entry: str,
    template: str,
    context_fn: Callable,
    sample_input_fn: Callable,
    task_fn: Callable,
):
    """Returns a string representation of the prompt for a given sample."""
    # TODO(geoffrey): figure out how to inject feature information into the prompt
    # TODO(geoffrey): figure out how to use {{x}} notation in the YAML file (probably needs regex)
    prompt = template.format(
        context=context_fn(entry), 
        ample_input=sample_input_fn(entry), 
        task=task_fn(entry),
    )
    return prompt


def get_context(
    entry: str,
    search_fn: Callable[[str], List[Dict[str, Any]]],
):
    """Returns a string representation of the context retrieved by `search_fn`."""
    k_samples = search_fn(query=entry)
    return json.dumps(k_samples, indent=2)


def get_sample_input(
    entry: str,
    input_col_name: str,
):
    """Returns a string representation of the sample input for the prompt."""
    return json.dumps({input_col_name: entry}, indent=2)


def get_task(entry: str, task_str: str):
    """Returns a string representation of the task for the prompt."""
    return task_str


def get_search_fn(retrieval_model: RetrievalModel, k: int) -> Callable[[str], List[Dict[str, Any]]]:
    """Returns a function that takes a query and returns the top k samples from the retrieval model."""
    return partial(_search_fn, retrieval_model=retrieval_model, k=k)


def _search_fn(
    query: str,
    retrieval_model: RetrievalModel,
    k: int,
) -> List[Dict[str, Any]]:
    return retrieval_model.search(query, k, return_data=True)


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
