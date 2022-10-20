from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ludwig.constants import DASK_MODULE_NAME
from ludwig.utils.types import DataFrame


def is_dask_lib(df_lib) -> bool:
    """Returns whether the dataframe library is dask."""
    return df_lib.__name__ == DASK_MODULE_NAME


def is_dask_backend(backend: Optional["Backend"]) -> bool:  # noqa: F821
    """Returns whether the backend's dataframe is dask."""
    return backend is not None and is_dask_lib(backend.df_engine.df_lib)


def is_dask_series_or_df(df: DataFrame, backend: Optional["Backend"]) -> bool:  # noqa: F821
    if is_dask_backend(backend):
        import dask.dataframe as dd

        return isinstance(df, dd.Series) or isinstance(df, dd.DataFrame)
    return False


def _get_shapes(row: pd.Series) -> Dict[str, Any]:
    return {k: np.array(v).shape for k, v in row.items()}


def _flatten(df_part: pd.DataFrame) -> pd.Series:
    for c in df_part.columns:
        df_part[c] = df_part[c].map(lambda x: np.array(x).reshape(-1))
    return df_part


def flatten_df(df: DataFrame, backend: "Backend") -> Tuple[DataFrame, Dict[str, Tuple]]:  # noqa: F821
    """Returns a flattened dataframe with a dictionary of the original shapes, keyed by dataframe columns."""
    # Workaround for: https://issues.apache.org/jira/browse/ARROW-5645
    df = backend.df_engine.persist(df)
    shapes_per_row = backend.df_engine.apply_objects(df, lambda row: _get_shapes(row))

    def reduce_fn(series):
        merged_shapes = None
        for shapes in series:
            if merged_shapes is None:
                merged_shapes = shapes.copy()
            else:
                merged_shapes = {k: max(v1, v2) for (k, v1), (_, v2) in zip(merged_shapes.items(), shapes.items())}
        return merged_shapes

    column_shapes = backend.df_engine.reduce_objects(shapes_per_row, reduce_fn)
    df = backend.df_engine.map_partitions(df, lambda x: _flatten(x))
    return df, column_shapes


def unflatten_df(df: DataFrame, column_shapes: Dict[str, Tuple], backend: "Backend") -> DataFrame:  # noqa: F821
    """Returns an unflattened dataframe, the reverse of flatten_df."""
    for c in df.columns:
        shape = column_shapes.get(c)
        if shape:
            df[c] = backend.df_engine.map_objects(df[c], lambda x: np.array(x).reshape(shape))
    return df


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


def set_index_name(pd_df: pd.DataFrame, name: str) -> pd.DataFrame:
    pd_df.index.name = name
    return pd_df
