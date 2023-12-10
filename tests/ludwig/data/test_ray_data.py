import os
import shutil
from unittest import mock

import pandas as pd
import pytest

# Skip these tests if Ray is not installed
ray = pytest.importorskip("ray")  # noqa
dask = pytest.importorskip("dask")  # noqa

from ludwig.data.dataset.ray import RayDatasetBatcher, read_remote_parquet  # noqa

# Mark the entire module as distributed
pytestmark = pytest.mark.distributed


def test_async_reader_error():
    pipeline = mock.Mock()
    features = {
        "num1": {"name": "num1", "type": "number"},
        "bin1": {"name": "bin1", "type": "binary"},
    }
    training_set_metadata = {
        "num1": {},
        "bin1": {},
    }

    # TODO: See if this is actually the correct fix for this test, not exactly sure what the test is trying to do
    with pytest.raises(AttributeError, match="'list_iterator' object has no attribute 'iter_batches'"):
        RayDatasetBatcher(
            dataset_epoch_iterator=iter([pipeline]),
            features=features,
            training_set_metadata=training_set_metadata,
            batch_size=64,
            samples_per_epoch=100,
            ignore_last=False,
        )


@pytest.fixture(scope="module")
def parquet_file(ray_cluster_2cpu) -> str:
    """Write a multi-file parquet dataset to the cwd.

    Returns:
        The path to the parquet dataset.
    """
    # The data needs to be written to a multi-file parquet format, otherwise the issue doesn't repro. To do this, we
    # partitition a test dataframe with dask and then write to file.
    df = pd.DataFrame({"col1": list(range(1000)), "col2": list(range(1000))})
    df = dask.dataframe.from_pandas(df, chunksize=100)

    # Typically we would write test data to a temporary directory, but the issue this was set up to test only happens
    # when using relative filepaths.
    cwd = os.getcwd()
    filepath = os.path.join(cwd, "data.training.parquet")
    df.to_parquet(filepath, engine="pyarrow")

    yield filepath

    # Clean up the data
    shutil.rmtree(filepath)


@pytest.fixture(scope="module", params=["absolute", "relative"])
def parquet_filepath(parquet_file: str, request: "pytest.FixtureRequest") -> str:
    """Convert a filepath in the CWD to either an absolute or relative path.

    Args:
        parquet_file: Absolute path to a parquet file in the CWD
        request: pytest request fixture with the fixture parameters

    Returns:
        Either the absolute or relative path of the parquet file.
    """
    filepath_type = request.param
    return parquet_file if filepath_type == "absolute" else os.path.basename(parquet_file)


def test_read_remote_parquet(parquet_filepath: str):
    """Test for the fix to https://github.com/ludwig-ai/ludwig/issues/3440.

    Parquet file reads will fail with `pyarrow.lib.ArrowInvalid` under the following conditions:
        1) The Parquet data is in multi-file format
        2) A relative filepath is passed to the read function
        3) A filesystem object is passed to the read function

    The issue can be resolved by either:
        1) Passing an absolute filepath
        2) Not passing a filesystem object
    """
    read_remote_parquet(parquet_filepath)
