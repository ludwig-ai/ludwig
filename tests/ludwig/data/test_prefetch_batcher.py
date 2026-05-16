"""Unit tests for RandomAccessBatcher prefetch mode and PandasDataset auto-prefetch.

Covers:
  - Prefetch produces identical batches to the sync path
  - All samples are covered across one and multiple epochs
  - set_epoch correctly resets state and re-starts the producer
  - last_batch() is accurate before and after exhaustion
  - Empty dataset terminates immediately
  - PandasDataset auto-enables prefetch when lazy columns are present
  - Audio lazy columns use a small max_workers to avoid CPU over-subscription
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from ludwig.data.batcher.random_access import RandomAccessBatcher
from ludwig.data.dataset.pandas import PandasDataset
from ludwig.data.lazy_utils import LazyColumn
from ludwig.data.sampler import DistributedSampler

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _SimpleDataset:
    """Minimal dataset backed by a single numpy array for batcher testing."""

    def __init__(self, n_samples: int, feature_name: str = "feat"):
        self.features = {feature_name: {"name": feature_name, "type": "number"}}
        self._data = {feature_name: np.arange(n_samples, dtype=np.float32)}
        self._n = n_samples
        self.feature_name = feature_name

    def get(self, col, idx):
        return self._data[col][idx]

    def __len__(self):
        return self._n


def _collect_batches(batcher) -> list[np.ndarray]:
    batches = []
    while not batcher.last_batch():
        batches.append(batcher.next_batch())
    return batches


def _make_batcher(n: int, batch_size: int, prefetch_size: int = 0, shuffle: bool = False) -> RandomAccessBatcher:
    ds = _SimpleDataset(n)
    sampler = DistributedSampler(n, shuffle=shuffle, random_seed=0)
    return RandomAccessBatcher(ds, sampler, batch_size=batch_size, prefetch_size=prefetch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prefetch_size", [0, 1, 2, 4])
def test_prefetch_yields_all_samples(prefetch_size):
    """Every sample must appear exactly once per epoch regardless of prefetch depth."""
    n, batch_size = 100, 16
    batcher = _make_batcher(n, batch_size, prefetch_size=prefetch_size)
    batches = _collect_batches(batcher)

    values = np.concatenate([b["feat"] for b in batches])
    assert len(values) == n
    assert set(values.tolist()) == set(range(n))


@pytest.mark.parametrize("prefetch_size", [1, 2, 4])
def test_prefetch_matches_sync_order(prefetch_size):
    """With shuffle=False, prefetch must return batches in the same order as sync."""
    n, batch_size = 50, 8
    sync_batcher = _make_batcher(n, batch_size, prefetch_size=0, shuffle=False)
    pre_batcher = _make_batcher(n, batch_size, prefetch_size=prefetch_size, shuffle=False)

    sync_values = np.concatenate([b["feat"] for b in _collect_batches(sync_batcher)])
    pre_values = np.concatenate([b["feat"] for b in _collect_batches(pre_batcher)])

    np.testing.assert_array_equal(sync_values, pre_values)


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_prefetch_exact_batch_sizes(prefetch_size):
    """All batches except possibly the last must be exactly batch_size samples."""
    n, batch_size = 55, 16
    batcher = _make_batcher(n, batch_size, prefetch_size=prefetch_size)
    batches = _collect_batches(batcher)

    for b in batches[:-1]:
        assert len(b["feat"]) == batch_size
    # Last batch may be smaller
    assert 0 < len(batches[-1]["feat"]) <= batch_size


# ─────────────────────────────────────────────────────────────────────────────
# last_batch() accuracy
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_last_batch_false_while_data_remains(prefetch_size):
    batcher = _make_batcher(20, 4, prefetch_size=prefetch_size)
    assert not batcher.last_batch()


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_last_batch_true_after_exhaustion(prefetch_size):
    batcher = _make_batcher(8, 4, prefetch_size=prefetch_size)
    _collect_batches(batcher)
    assert batcher.last_batch()


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_next_batch_raises_on_exhaustion(prefetch_size):
    batcher = _make_batcher(4, 4, prefetch_size=prefetch_size)
    _collect_batches(batcher)
    with pytest.raises(StopIteration):
        batcher.next_batch()


# ─────────────────────────────────────────────────────────────────────────────
# set_epoch
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prefetch_size", [0, 2, 4])
def test_set_epoch_allows_multiple_passes(prefetch_size):
    """set_epoch must reset state so the full dataset is available again."""
    n, batch_size, n_epochs = 30, 8, 3
    batcher = _make_batcher(n, batch_size, prefetch_size=prefetch_size)
    all_samples = []
    for epoch in range(n_epochs):
        batcher.set_epoch(epoch, batch_size)
        epoch_samples = np.concatenate([b["feat"] for b in _collect_batches(batcher)])
        assert len(epoch_samples) == n, f"epoch {epoch}: got {len(epoch_samples)}, expected {n}"
        all_samples.append(epoch_samples)

    # All epochs must cover the full dataset
    for epoch_samples in all_samples:
        assert set(epoch_samples.tolist()) == set(range(n))


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_set_epoch_batch_size_change(prefetch_size):
    """set_epoch must correctly apply a new batch_size for the next epoch."""
    batcher = _make_batcher(40, 8, prefetch_size=prefetch_size)
    _collect_batches(batcher)  # exhaust epoch 0

    new_batch_size = 4
    batcher.set_epoch(1, new_batch_size)
    batches = _collect_batches(batcher)
    assert batcher.batch_size == new_batch_size
    assert all(len(b["feat"]) == new_batch_size for b in batches)
    assert sum(len(b["feat"]) for b in batches) == 40


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_empty_dataset_last_batch_immediately(prefetch_size):
    """A dataset with 0 samples must signal last_batch() True immediately."""
    batcher = _make_batcher(0, 4, prefetch_size=prefetch_size)
    assert batcher.last_batch()


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_single_sample_dataset(prefetch_size):
    """Dataset with exactly 1 sample must produce exactly 1 batch."""
    batcher = _make_batcher(1, 4, prefetch_size=prefetch_size)
    batches = _collect_batches(batcher)
    assert len(batches) == 1
    assert len(batches[0]["feat"]) == 1


@pytest.mark.parametrize("prefetch_size", [0, 2])
def test_batch_size_larger_than_dataset(prefetch_size):
    """When batch_size > n_samples, exactly one batch with all samples is produced."""
    n, batch_size = 5, 32
    batcher = _make_batcher(n, batch_size, prefetch_size=prefetch_size)
    batches = _collect_batches(batcher)
    assert len(batches) == 1
    assert len(batches[0]["feat"]) == n


# ─────────────────────────────────────────────────────────────────────────────
# PandasDataset auto-prefetch
# ─────────────────────────────────────────────────────────────────────────────


def test_pandas_dataset_auto_prefetch_for_lazy_columns():
    """PandasDataset must auto-enable prefetch when lazy columns are present."""
    n = 20
    paths = np.array([f"/fake/{i}.wav" for i in range(n)], dtype=object)
    decode_fn = lambda p: np.zeros((8, 23), dtype=np.float32)
    decode_fn.__name__ = "mock_decode"

    proc_col = "audio"
    features = {proc_col: {"name": "audio_feat", "type": "audio"}}
    training_set_metadata = {
        "audio_feat": {
            "lazy": True,
            "reshape": None,
            "lazy_audio_params": {
                "audio_feature_dict": {"type": "fbank"},
                "feature_dim": 8,
                "max_length": 23,
                "padding_value": 0.0,
                "normalization_type": None,
            },
        }
    }

    with patch("ludwig.features.audio_feature.AudioFeatureMixin._make_lazy_decode_fn", return_value=decode_fn):
        ds = PandasDataset(
            {proc_col: paths},
            features,
            data_cache_fp=None,
            training_set_metadata=training_set_metadata,
        )

    assert ds._has_lazy_columns(), "Dataset with audio paths should have lazy columns"

    # initialize_batcher should auto-enable prefetch (prefetch_size > 0)
    with ds.initialize_batcher(batch_size=4, should_shuffle=False) as batcher:
        assert batcher._prefetch_size > 0, f"Expected prefetch_size > 0 for lazy dataset, got {batcher._prefetch_size}"
        batches = _collect_batches(batcher)

    assert sum(len(b[proc_col]) for b in batches) == n


def test_pandas_dataset_no_auto_prefetch_for_eager_columns():
    """PandasDataset must NOT enable prefetch for purely eager (pre-decoded) columns."""
    n = 20
    proc_col = "num"
    features = {proc_col: {"name": "num_feat", "type": "number"}}
    training_set_metadata = {"num_feat": {"lazy": False}}
    ds = PandasDataset(
        {proc_col: np.arange(n, dtype=np.float32)},
        features,
        data_cache_fp=None,
        training_set_metadata=training_set_metadata,
    )

    assert not ds._has_lazy_columns()

    with ds.initialize_batcher(batch_size=4, should_shuffle=False) as batcher:
        assert batcher._prefetch_size == 0, (
            f"Expected prefetch_size == 0 for eager dataset, got {batcher._prefetch_size}"
        )


def test_pandas_dataset_explicit_prefetch_override():
    """Callers can override auto-prefetch by passing prefetch_size explicitly."""
    n = 10
    proc_col = "num"
    features = {proc_col: {"name": "num_feat", "type": "number"}}
    training_set_metadata = {"num_feat": {"lazy": False}}
    ds = PandasDataset(
        {proc_col: np.arange(n, dtype=np.float32)},
        features,
        data_cache_fp=None,
        training_set_metadata=training_set_metadata,
    )

    # Explicit prefetch_size=3 must be respected even on an eager dataset
    with ds.initialize_batcher(batch_size=4, should_shuffle=False, prefetch_size=3) as batcher:
        assert batcher._prefetch_size == 3


# ─────────────────────────────────────────────────────────────────────────────
# Audio max_workers: avoid CPU over-subscription
# ─────────────────────────────────────────────────────────────────────────────


def test_audio_lazy_column_max_workers_capped():
    """LazyColumn created for audio must use max_workers ≤ cpu_count // torch_threads.

    The FBANK decode uses PyTorch's internal thread pool.  Using many workers
    in parallel creates CPU over-subscription and can be up to 5× slower than
    a single worker.  This test verifies the cap is respected.
    """
    import torch

    n = 10
    paths = np.array([f"/fake/{i}.wav" for i in range(n)], dtype=object)
    decode_fn = lambda p: np.zeros((8, 23), dtype=np.float32)
    decode_fn.__name__ = "mock_decode"

    proc_col = "audio"
    features = {proc_col: {"name": "audio_feat", "type": "audio"}}
    training_set_metadata = {
        "audio_feat": {
            "lazy": True,
            "reshape": None,
            "lazy_audio_params": {
                "audio_feature_dict": {"type": "fbank"},
                "feature_dim": 8,
                "max_length": 23,
                "padding_value": 0.0,
                "normalization_type": None,
            },
        }
    }

    with patch("ludwig.features.audio_feature.AudioFeatureMixin._make_lazy_decode_fn", return_value=decode_fn):
        ds = PandasDataset(
            {proc_col: paths},
            features,
            data_cache_fp=None,
            training_set_metadata=training_set_metadata,
        )

    lazy_col = ds.dataset[proc_col]
    assert isinstance(lazy_col, LazyColumn), "Expected LazyColumn for audio"

    cpu_count = os.cpu_count() or 4
    torch_threads = max(1, torch.get_num_threads())
    max_safe_workers = max(1, cpu_count // torch_threads)

    assert lazy_col._max_workers <= max_safe_workers, (
        f"Audio LazyColumn max_workers={lazy_col._max_workers} exceeds safe limit "
        f"{max_safe_workers} (cpu_count={cpu_count}, torch_threads={torch_threads}). "
        "Over-subscribing CPUs causes severe FBANK decode slowdowns."
    )
