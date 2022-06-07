import numpy as np
import pandas as pd

from ludwig.backend import Backend, LOCAL_BACKEND
from ludwig.constants import DASK_MODULE_NAME


def is_dask_lib(df_lib):
    """Returns whether the dataframe library is dask."""
    return df_lib.__name__ == DASK_MODULE_NAME


def is_dask_backend(backend: Backend):
    """Returns whether the backend's dataframe is dask."""
    return backend.df_engine.df_lib.__name__ == DASK_MODULE_NAME


def flatten_df(df, backend):
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


def unflatten_df(df, column_shapes, backend):
    for c in df.columns:
        shape = column_shapes.get(c)
        if shape:
            df[c] = backend.df_engine.map_objects(df[c], lambda x: np.array(x).reshape(shape))
    return df


def to_numpy_dataset(df, backend=LOCAL_BACKEND):
    dataset = {}
    for col in df.columns:
        res = df[col]
        if is_dask_backend(backend):
            res = res.compute()
        dataset[col] = np.stack(res.to_numpy())
    return dataset


def from_numpy_dataset(dataset):
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
