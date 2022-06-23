from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ludwig.constants import DASK_MODULE_NAME
from ludwig.datasets.base_dataset import BaseDataset
from ludwig.utils.types import DataFrame


def is_dask_lib(df_lib) -> bool:
    """Returns whether the dataframe library is dask."""
    return df_lib.__name__ == DASK_MODULE_NAME


def is_dask_backend(backend: "Backend") -> bool:  # noqa: F821
    """Returns whether the backend's dataframe is dask."""
    return is_dask_lib(backend.df_engine.df_lib)


def flatten_df(df: DataFrame, backend: "Backend") -> Tuple[DataFrame, Dict[str, Tuple]]:  # noqa: F821
    """Returns a flattened dataframe with a dictionary of the original shapes, keyed by dataframe columns."""
    # Workaround for: https://issues.apache.org/jira/browse/ARROW-5645
    column_shapes = {}
    for c in df.columns:
        df = backend.df_engine.persist(df)
        shape = backend.df_engine.compute(
            backend.df_engine.map_objects(
                df[c],
                lambda x: np.array(x).shape,
            ).max()
        )

        if len(shape) > 1:
            column_shapes[c] = shape
            df[c] = backend.df_engine.map_objects(df[c], lambda x: np.array(x).reshape(-1))
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
        dataset[col] = np.stack(res.to_numpy())
    return dataset


def from_numpy_dataset(dataset: BaseDataset) -> pd.DataFrame:
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
