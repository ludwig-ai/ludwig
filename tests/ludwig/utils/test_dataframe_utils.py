import numpy as np
import pandas as pd
import pytest

from ludwig.backend import create_backend, LOCAL_BACKEND
from ludwig.utils.dataframe_utils import to_numpy_dataset

try:
    import dask.dataframe as dd
except ImportError:
    pass


@pytest.mark.distributed
def test_to_numpy_dataset_with_dask(ray_cluster_2cpu):
    dd_df = dd.from_pandas(pd.DataFrame([[1, 2, 3]], columns=["col1", "col2", "col3"]), npartitions=1)
    ray_backend = create_backend("ray")

    np_df = to_numpy_dataset(dd_df, backend=ray_backend)

    assert np_df == {"col1": np.array([1]), "col2": np.array([2]), "col3": np.array([3])}


@pytest.mark.distributed
def test_to_numpy_dataset_with_dask_backend_mismatch():
    dd_df = dd.from_pandas(pd.DataFrame([[1, 2, 3]], columns=["col1", "col2", "col3"]), npartitions=1)

    with pytest.raises(AttributeError):
        to_numpy_dataset(dd_df, backend=LOCAL_BACKEND)


def test_to_numpy_dataset_with_pandas():
    pd_df = pd.DataFrame([[1, 2, 3]], columns=["col1", "col2", "col3"])

    np_df = to_numpy_dataset(pd_df, backend=LOCAL_BACKEND)

    assert np_df == {"col1": np.array([1]), "col2": np.array([2]), "col3": np.array([3])}


def test_to_numpy_dataset_empty_with_columns():
    pd_df = pd.DataFrame(columns=["col1", "col2", "col3"])

    np_df = to_numpy_dataset(pd_df, backend=LOCAL_BACKEND)

    assert np_df == {"col1": [], "col2": [], "col3": []}


def test_to_numpy_dataset_empty():
    pd_df = pd.DataFrame()

    np_df = to_numpy_dataset(pd_df, backend=LOCAL_BACKEND)

    assert np_df == {}


@pytest.mark.distributed
def test_to_numpy_dataset_with_pandas_backend_mismatch(ray_cluster_2cpu):
    pd_df = pd.DataFrame([[1, 2, 3]], columns=["col1", "col2", "col3"])
    ray_backend = create_backend("ray")

    with pytest.raises(AttributeError):
        to_numpy_dataset(pd_df, backend=ray_backend)
