import os
import shutil
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Skip these tests if Ray is not installed
ray = pytest.importorskip("ray")
dask = pytest.importorskip("dask")

from ludwig.data.dataset.ray import RayDataset, RayDatasetBatcher, RayDatasetShardBatcher, read_remote_parquet  # noqa

# Mark the entire module as distributed
pytestmark = [pytest.mark.distributed, pytest.mark.distributed_d]


def test_prefetch_batches_value():
    """Regression test: both Ray batcher async readers must use prefetch_batches > 1.

    prefetch_batches=1 starves the GPU because only one batch is in flight at a time.
    The fix sets it to 4, which keeps more batches queued so the GPU stays busy.
    See https://github.com/ludwig-ai/ludwig/issues/4142
    """
    import inspect
    import re

    src_batcher = inspect.getsource(RayDatasetBatcher._create_async_reader)
    src_shard = inspect.getsource(RayDatasetShardBatcher._create_async_reader)

    def _get_prefetch_value(src: str) -> int:
        match = re.search(r"prefetch_batches\s*=\s*(\d+)", src)
        assert match, "prefetch_batches kwarg not found in async reader source"
        return int(match.group(1))

    assert _get_prefetch_value(src_batcher) > 1, (
        "RayDatasetBatcher uses prefetch_batches=1, which starves the GPU. Increase it."
    )
    assert _get_prefetch_value(src_shard) > 1, (
        "RayDatasetShardBatcher uses prefetch_batches=1, which starves the GPU. Increase it."
    )


def test_train_fn_passes_device_to_remote_trainer():
    """Regression test: train_fn must pass device= to RemoteTrainer so the trainer's self.device
    matches the Ray-assigned GPU rather than falling back to get_torch_device().

    Without this, metrics_to_device() and batch-to-device tensors may disagree with the model's
    actual placement when running inside a Ray Train worker.
    See https://github.com/ludwig-ai/ludwig/issues/4142
    """
    import inspect

    from ludwig.backend.ray import train_fn

    src = inspect.getsource(train_fn)
    # The RemoteTrainer() call must forward device= so Trainer.__init__ doesn't
    # silently override it with get_torch_device().
    assert "RemoteTrainer(model=model, device=" in src, (
        "train_fn must pass device= to RemoteTrainer to avoid device mismatch. "
        "See https://github.com/ludwig-ai/ludwig/issues/4142"
    )


def test_progress_bar_does_not_call_rt_report_per_batch():
    """Regression test: LudwigProgressBar must not call rt.report() on every training batch.

    Each rt.report() call requires a round-trip through the Ray GCS (~1.9 s). With hundreds of
    batches per run this completely dominates wall-clock time.  The fix silently suppresses the
    tqdm bar inside Ray workers instead of reporting per-batch progress via rt.report().
    """
    from ludwig.progress_bar import LudwigProgressBar

    with mock.patch("ludwig.progress_bar.rt") as mock_rt:
        pbar = LudwigProgressBar(report_to_ray=True, config={"total": 10, "desc": "test"}, is_coordinator=True)
        for _ in range(10):
            pbar.update(1)
        pbar.close()

    mock_rt.report.assert_not_called()


def test_async_reader_error():
    """Test that RayDatasetBatcher handles a dataset that produces no batches.

    When iter_batches yields nothing, the batcher should end up with last_batch=True.
    """
    mock_dataset = mock.Mock()
    # iter_batches yields nothing (empty dataset)
    mock_dataset.iter_batches.return_value = iter([])

    features = {
        "num1": {"name": "num1", "type": "number"},
        "bin1": {"name": "bin1", "type": "binary"},
    }
    training_set_metadata = {
        "num1": {},
        "bin1": {},
    }

    batcher = RayDatasetBatcher(
        dataset=mock_dataset,
        features=features,
        training_set_metadata=training_set_metadata,
        batch_size=64,
        samples_per_epoch=100,
    )
    # With no data to read, the batcher should immediately signal last batch
    assert batcher.last_batch()


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


# ---------------------------------------------------------------------------
# Lazy decode tests
# ---------------------------------------------------------------------------


def _make_ray_dataset(df: pd.DataFrame, features: dict, training_set_metadata: dict) -> RayDataset:
    """Construct a RayDataset directly from a pandas DataFrame for testing."""
    ds = RayDataset.__new__(RayDataset)
    ds.ds = ray.data.from_pandas(df)
    ds.features = features
    ds.training_set_metadata = training_set_metadata
    ds.data_cache_fp = None
    ds.data_parquet_fp = None
    return ds


def test_with_lazy_decode_audio(ray_cluster_2cpu, tmp_path):
    """_with_lazy_decode must replace path strings with decoded numpy arrays for audio features.

    Before the fix, the Ray backend passed raw path strings to workers; the batcher then tried
    to np.stack strings which either hung or produced garbage.  After the fix, map_batches
    decode transforms run inside each worker and return proper numpy arrays.

    The decode function is mocked here so the test has no soundfile dependency.
    The real audio decode pipeline is exercised by test_ray_audio_basic (integration test).
    """
    from unittest.mock import patch

    proc_col = "audio_proc_col"
    feature_name = "audio_0"
    feature_dim = 8
    max_length = 5

    dummy_array = np.zeros((feature_dim, max_length), dtype=np.float32)

    def _mock_make_lazy_decode_fn(**kwargs):
        def decode(path):
            return dummy_array

        decode.__name__ = "audio_lazy_decode"
        return decode

    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "audio"}}
    training_set_metadata = {
        feature_name: {
            "lazy": True,
            "reshape": None,
            "lazy_audio_params": {
                "audio_feature_dict": {"type": "fbank"},
                "feature_dim": feature_dim,
                "max_length": max_length,
                "padding_value": 0.0,
                "normalization_type": None,
            },
        }
    }

    df = pd.DataFrame({proc_col: ["/fake/path/a.wav", "/fake/path/b.wav"]})
    ray_ds = _make_ray_dataset(df, features, training_set_metadata)

    # Before: raw dataset has path strings, not arrays.
    raw_batch = ray_ds.ds.take_batch(batch_size=2, batch_format="pandas")
    assert isinstance(raw_batch[proc_col].iloc[0], str), "Expected path string in raw dataset"

    # After: _with_lazy_decode adds a map_batches transform that decodes paths to arrays.
    with patch("ludwig.features.audio_feature.AudioFeatureMixin._make_lazy_decode_fn", _mock_make_lazy_decode_fn):
        decoded_ds = ray_ds._with_lazy_decode(ray_ds.ds)

    decoded_batch = decoded_ds.take_batch(batch_size=2, batch_format="pandas")
    first = decoded_batch[proc_col].iloc[0]
    assert isinstance(first, np.ndarray), f"Expected numpy array after decode, got {type(first)}"
    assert first.shape == (feature_dim, max_length), f"Expected shape ({feature_dim}, {max_length}), got {first.shape}"


def test_with_lazy_decode_image(ray_cluster_2cpu, tmp_path):
    """_with_lazy_decode must replace path strings with decoded numpy arrays for image features.

    Same root cause as audio: without decode transforms, workers receive file paths instead
    of tensors, causing hangs or crashes in the batcher.
    """
    PIL = pytest.importorskip("PIL.Image")

    # Write a tiny PNG (12×12 RGB).
    img_path = str(tmp_path / "test.png")
    img = PIL.fromarray(np.zeros((12, 12, 3), dtype=np.uint8))
    img.save(img_path)

    proc_col = "image_proc_col"
    feature_name = "image_0"
    h, w, c = 12, 12, 3

    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "image"}}
    training_set_metadata = {
        feature_name: {
            "lazy": True,
            "reshape": None,
            "lazy_image_params": {
                "img_width": w,
                "img_height": h,
                "should_resize": False,
                "num_channels": c,
                "resize_method": "interpolate",
                "user_specified_num_channels": True,
                "standardize_image": "pixel_normalization",
                "channel_class_map": [],
                "default_image_shape": [c, h, w],
            },
        }
    }

    df = pd.DataFrame({proc_col: [img_path, img_path]})
    ray_ds = _make_ray_dataset(df, features, training_set_metadata)

    # Before: raw dataset has path strings.
    raw_batch = ray_ds.ds.take_batch(batch_size=2, batch_format="pandas")
    assert isinstance(raw_batch[proc_col].iloc[0], str), "Expected path string in raw dataset"

    # After: _with_lazy_decode adds a map_batches transform that decodes paths to arrays.
    decoded_ds = ray_ds._with_lazy_decode(ray_ds.ds)
    decoded_batch = decoded_ds.take_batch(batch_size=2, batch_format="pandas")
    first = decoded_batch[proc_col].iloc[0]
    assert isinstance(first, np.ndarray), f"Expected numpy array after decode, got {type(first)}"
    assert first.ndim == 3, f"Expected 3-D image array (C × H × W), got shape {first.shape}"
    assert first.shape == (c, h, w), f"Expected shape {(c, h, w)}, got {first.shape}"


def test_with_lazy_decode_non_lazy_passthrough(ray_cluster_2cpu):
    """_with_lazy_decode must not touch features that are not lazy."""
    proc_col = "num_col"
    feature_name = "num_0"

    features = {proc_col: {"name": feature_name, "column": feature_name, "type": "number"}}
    training_set_metadata = {feature_name: {"lazy": False}}

    df = pd.DataFrame({proc_col: [1.0, 2.0, 3.0]})
    ray_ds = _make_ray_dataset(df, features, training_set_metadata)

    decoded_ds = ray_ds._with_lazy_decode(ray_ds.ds)

    # Dataset should be identical — no transforms added.
    batch = decoded_ds.take_batch(batch_size=3, batch_format="pandas")
    assert list(batch[proc_col]) == [1.0, 2.0, 3.0]
