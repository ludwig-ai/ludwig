"""Unit tests for Trainer helper methods extracted in PR-6.

Tests use lightweight mocks so they run without a GPU, a trained model, or a
full dataset — each extracted method is tested in isolation.
"""

from unittest.mock import MagicMock

import numpy as np
import torch


class TestBatchToTensors:
    """Tests for Trainer._batch_to_tensors."""

    def _make_trainer(self, device="cpu"):
        """Build a minimal Trainer-like object with just the state _batch_to_tensors needs."""
        from ludwig.trainers.trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.device = device

        # Create mock input and output features
        in_feat = MagicMock()
        in_feat.feature_name = "text"
        in_feat.proc_column = "text_proc"

        out_feat = MagicMock()
        out_feat.feature_name = "label"
        out_feat.proc_column = "label_proc"

        trainer.model = MagicMock()
        trainer.model.input_features = {"text": in_feat}
        trainer.model.output_features = {"label": out_feat}
        return trainer

    def test_returns_inputs_and_targets(self):
        trainer = self._make_trainer()
        batch = {
            "text_proc": np.array([1.0, 2.0, 3.0]),
            "label_proc": np.array([0.0, 1.0, 0.0]),
        }
        inputs, targets = trainer._batch_to_tensors(batch)
        assert "text" in inputs
        assert "label" in targets

    def test_inputs_are_tensors(self):
        trainer = self._make_trainer()
        batch = {
            "text_proc": np.array([1.0, 2.0]),
            "label_proc": np.array([0.0, 1.0]),
        }
        inputs, targets = trainer._batch_to_tensors(batch)
        assert isinstance(inputs["text"], torch.Tensor)
        assert isinstance(targets["label"], torch.Tensor)

    def test_values_match_numpy_input(self):
        trainer = self._make_trainer()
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        batch = {"text_proc": arr, "label_proc": np.zeros(5)}
        inputs, _ = trainer._batch_to_tensors(batch)
        np.testing.assert_array_equal(inputs["text"].numpy(), arr)

    def test_tensors_on_correct_device(self):
        trainer = self._make_trainer(device="cpu")
        batch = {"text_proc": np.array([1.0]), "label_proc": np.array([0.0])}
        inputs, targets = trainer._batch_to_tensors(batch)
        assert inputs["text"].device.type == "cpu"
        assert targets["label"].device.type == "cpu"

    def test_multiple_input_features(self):
        from ludwig.trainers.trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.device = "cpu"

        feats = {}
        for name in ("a", "b", "c"):
            f = MagicMock()
            f.feature_name = name
            f.proc_column = f"{name}_proc"
            feats[name] = f

        trainer.model = MagicMock()
        trainer.model.input_features = feats
        trainer.model.output_features = {}

        batch = {f"{n}_proc": np.array([float(i)]) for i, n in enumerate("abc")}
        inputs, targets = trainer._batch_to_tensors(batch)
        assert set(inputs.keys()) == {"a", "b", "c"}
        assert targets == {}
