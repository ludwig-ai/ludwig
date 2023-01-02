from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DASK_MODULE_NAME
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.utils.types import DataFrame


@DeveloperAPI
def is_dask_lib(df_lib) -> bool:
    """Returns whether the dataframe library is dask."""
    return df_lib.__name__ == DASK_MODULE_NAME


@DeveloperAPI
def is_dask_backend(backend: Optional["Backend"]) -> bool:  # noqa: F821
    """Returns whether the backend's dataframe is dask."""
    return backend is not None and is_dask_lib(backend.df_engine.df_lib)


@DeveloperAPI
def is_dask_series_or_df(df: DataFrame, backend: Optional["Backend"]) -> bool:  # noqa: F821
    if is_dask_backend(backend):
        import dask.dataframe as dd

        return isinstance(df, dd.Series) or isinstance(df, dd.DataFrame)
    return False


@DeveloperAPI
def flatten_df(df: DataFrame, df_engine: DataFrameEngine) -> Tuple[DataFrame, Dict[str, Tuple]]:  # noqa: F821
    """Returns a flattened dataframe with a dictionary of the original shapes, keyed by dataframe columns."""
    # Workaround for: https://issues.apache.org/jira/browse/ARROW-5645
    column_shapes = {}
    for c in df.columns:
        df = df_engine.persist(df)
        shape = df_engine.compute(
            df_engine.map_objects(
                df[c],
                lambda x: np.array(x).shape,
            ).max()
        )

        if len(shape) > 1:
            column_shapes[c] = shape
            df[c] = df_engine.map_objects(df[c], lambda x: np.array(x).reshape(-1))
    return df, column_shapes


@DeveloperAPI
def unflatten_df(df: DataFrame, column_shapes: Dict[str, Tuple], df_engine: DataFrameEngine) -> DataFrame:  # noqa: F821
    """Returns an unflattened dataframe, the reverse of flatten_df."""
    for c in df.columns:
        shape = column_shapes.get(c)
        if shape:
            df[c] = df_engine.map_objects(df[c], lambda x: np.array(x).reshape(shape))
    return df


@DeveloperAPI
def to_numpy_dataset(df: DataFrame, backend: Optional["Backend"] = None) -> Dict[str, np.ndarray]:  # noqa: F821
    """Returns a dictionary of numpy arrays, keyed by the columns of the given dataframe."""
    dataset = {}
    for col in df.columns:
        res = df[col]
        if backend and is_dask_backend(backend):
            res = res.compute()
        if len(df.index) != 0:
            dataset[col] = np.stack(res.to_numpy())
        else:
            # Dataframe is empty.
            # Use to_list() directly, as np.stack() requires at least one array to stack.
            dataset[col] = res.to_list()
    return dataset


@DeveloperAPI
def from_numpy_dataset(dataset) -> pd.DataFrame:
    """Returns a pandas dataframe from the dataset."""
    col_mapping = {}
    for k, v in dataset.items():
        if len(v.shape) > 1:
            # unstacking, needed for ndarrays of dimension 2 and more
            (*vals,) = v
        else:
            # not unstacking. Needed because otherwise pandas casts types
            # the way it wants, like converting a list of float32 scalats
            # to a column of float64
            vals = v
        col_mapping[k] = vals
    return pd.DataFrame.from_dict(col_mapping)


@DeveloperAPI
def set_index_name(pd_df: pd.DataFrame, name: str) -> pd.DataFrame:
    pd_df.index.name = name
    return pd_df


@DeveloperAPI
def to_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    return [df[i : i + batch_size].copy() for i in range(0, df.shape[0], batch_size)]


@DeveloperAPI
def from_batches(batches: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(batches)
