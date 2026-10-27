"""Tests for the Ray worker -> driver checkpoint handoff.

The driver must never unpickle files from a Ray checkpoint directory, since that directory may live on shared storage
writable by other principals (GHSA-p3h5-5763-3r3p).
"""

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ray")

from ludwig.backend.ray import (
    _EVAL_METRICS_FILE,
    _read_json,
    _read_train_checkpoint,
    _write_json_atomically,
    _write_train_checkpoint,
)
from ludwig.utils.metric_utils import TrainerMetric

# Files that older Ludwig versions loaded with torch.load(weights_only=False).
LEGACY_PICKLE_FILES = ("train_other.pt", "train_results.pt", "eval_results.pt")


class _MarkerGadget:
    """Pickle payload that creates a marker file when unpickled."""

    def __init__(self, marker_path: Path):
        self.marker_path = marker_path

    def __reduce__(self):
        return (Path.touch, (self.marker_path,))


def _plant_pickle_gadgets(checkpoint_dir: Path, marker_path: Path) -> None:
    for file_name in LEGACY_PICKLE_FILES:
        with open(checkpoint_dir / file_name, "wb") as f:
            pickle.dump(_MarkerGadget(marker_path), f)


def test_marker_gadget_fires_when_unpickled(tmp_path):
    """Positive control: the planted files do execute code if anything unpickles them."""
    marker_path = tmp_path / "pwned"
    _plant_pickle_gadgets(tmp_path, marker_path)

    with open(tmp_path / "train_other.pt", "rb") as f:
        pickle.load(f)

    assert marker_path.exists()


def _sample_trainer_metrics():
    metrics = defaultdict(lambda: defaultdict(list))
    metrics["label"]["accuracy"].append(TrainerMetric(epoch=1, step=10, value=np.float32(0.75)))
    metrics["combined"]["loss"].append(TrainerMetric(epoch=1, step=10, value=torch.tensor(0.5)))
    return metrics


def test_train_checkpoint_round_trip(tmp_path):
    state_dict = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    other_results = [_sample_trainer_metrics(), {}, {}]

    _write_train_checkpoint(str(tmp_path), state_dict, other_results, "combined", "loss")
    loaded_state_dict, loaded_other_results, val_field, val_metric = _read_train_checkpoint(str(tmp_path))

    assert torch.equal(loaded_state_dict["weight"], state_dict["weight"])
    assert (val_field, val_metric) == ("combined", "loss")
    assert loaded_other_results[0]["label"]["accuracy"] == [TrainerMetric(epoch=1, step=10, value=0.75)]
    assert loaded_other_results[0]["combined"]["loss"] == [TrainerMetric(epoch=1, step=10, value=0.5)]
    assert isinstance(loaded_other_results[0]["combined"]["loss"][0], TrainerMetric)
    assert loaded_other_results[1:] == [{}, {}]
    assert not any(name.endswith(".pt") for name in os.listdir(tmp_path))


def test_train_checkpoint_ignores_planted_pickle(tmp_path):
    marker_path = tmp_path / "pwned"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_train_checkpoint(str(checkpoint_dir), {"weight": torch.zeros(1)}, [{}, {}, {}], "combined", "loss")
    _plant_pickle_gadgets(checkpoint_dir, marker_path)

    _read_train_checkpoint(str(checkpoint_dir))

    assert not marker_path.exists()


def test_train_checkpoint_without_json_fails_instead_of_unpickling(tmp_path):
    """A checkpoint containing only pickle files (the legacy or attacker-crafted layout) is rejected."""
    marker_path = tmp_path / "pwned"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _plant_pickle_gadgets(checkpoint_dir, marker_path)

    with pytest.raises(FileNotFoundError):
        _read_train_checkpoint(str(checkpoint_dir))

    assert not marker_path.exists()


def test_eval_metrics_round_trip(tmp_path):
    eval_metrics = {"label": {"accuracy": np.float64(0.9), "confusion": np.array([[1, 2], [3, 4]])}}
    path = str(tmp_path / _EVAL_METRICS_FILE)

    _write_json_atomically(eval_metrics, path)

    assert _read_json(path) == {"label": {"accuracy": 0.9, "confusion": [[1, 2], [3, 4]]}}


def test_eval_metrics_without_json_fails_instead_of_unpickling(tmp_path):
    marker_path = tmp_path / "pwned"
    _plant_pickle_gadgets(tmp_path, marker_path)

    with pytest.raises(FileNotFoundError):
        _read_json(str(tmp_path / _EVAL_METRICS_FILE))

    assert not marker_path.exists()


def test_unserializable_result_fails_without_writing_partial_file(tmp_path):
    path = tmp_path / _EVAL_METRICS_FILE

    with pytest.raises(TypeError, match="object"):
        _write_json_atomically({"label": {"accuracy": object()}}, str(path))

    assert not path.exists()
