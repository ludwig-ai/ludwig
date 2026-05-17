"""Unit tests for Predictor.batch_collect_activations.

Verifies the per-batch CPU-offload accumulation pattern that avoids OOM when
collecting activations from large models over many batches.
"""

import contextlib
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _make_predictor(batches, model_outputs_per_batch):
    """Build a Predictor with mocked internals.

    Args:
        batches: list of batch dicts to return from batcher.next_batch()
        model_outputs_per_batch: list of dicts (one per batch) returned by _predict_on_inputs
    """
    from ludwig.distributed.base import LocalStrategy
    from ludwig.models.base import BaseModel
    from ludwig.models.predictor import Predictor

    # Minimal mock BaseModel that satisfies isinstance check
    mock_model = MagicMock(spec=BaseModel)
    in_feat = MagicMock()
    in_feat.feature_name = "text"
    in_feat.proc_column = "text_proc"
    mock_model.input_features = {"text": in_feat}

    mock_dist_model = MagicMock()
    mock_dist_model.training = False

    predictor = Predictor.__new__(Predictor)
    predictor._batch_size = 4
    predictor._distributed = LocalStrategy()
    predictor.report_tqdm_to_ray = False
    predictor.device = "cpu"
    predictor.dist_model = mock_dist_model
    predictor.model = mock_model

    # Build batcher mock that yields batches one by one
    call_count = [0]

    def next_batch():
        b = batches[call_count[0]]
        call_count[0] += 1
        return b

    def last_batch():
        return call_count[0] >= len(batches)

    batcher = MagicMock()
    batcher.next_batch.side_effect = next_batch
    batcher.last_batch.side_effect = last_batch
    batcher.steps_per_epoch = len(batches)

    # Wrap initialize_batcher as a context manager
    @contextlib.contextmanager
    def init_batcher(*args, **kwargs):
        yield batcher

    mock_model.metrics_to_device = MagicMock()

    # Patch dataset
    predictor._dataset_mock = MagicMock()
    predictor._dataset_mock.initialize_batcher = init_batcher

    # Patch _predict_on_inputs to return successive outputs
    output_iter = iter(model_outputs_per_batch)
    predictor._predict_on_inputs = lambda inputs: next(output_iter)

    return predictor


class TestBatchCollectActivationsCPUOffload:
    def test_tensors_concatenated_across_batches(self):
        batches = [{"text_proc": np.zeros(4)} for _ in range(3)]
        per_batch = [
            {"hidden": torch.ones(4, 8)},
            {"hidden": torch.ones(4, 8) * 2},
            {"hidden": torch.ones(4, 8) * 3},
        ]
        predictor = _make_predictor(batches, per_batch)
        result = predictor.batch_collect_activations(layer_names=["hidden"], dataset=predictor._dataset_mock)
        name, tensor = result[0]
        assert name == "hidden"
        assert tensor.shape == (12, 8)  # 3 batches × 4 rows

    def test_batch1_values_in_concatenated_output(self):
        batches = [{"text_proc": np.zeros(2)} for _ in range(2)]
        per_batch = [
            {"out": torch.full((2, 3), 1.0)},
            {"out": torch.full((2, 3), 9.0)},
        ]
        predictor = _make_predictor(batches, per_batch)
        result = predictor.batch_collect_activations(layer_names=["out"], dataset=predictor._dataset_mock)
        _, tensor = result[0]
        assert tensor[0, 0].item() == pytest.approx(1.0)
        assert tensor[2, 0].item() == pytest.approx(9.0)

    def test_output_tensors_are_on_cpu(self):
        """Result tensors must be on CPU regardless of where they were produced."""
        batches = [{"text_proc": np.zeros(2)} for _ in range(2)]
        per_batch = [
            {"layer": torch.ones(2, 4)},  # already CPU in test; accumulation must stay CPU
            {"layer": torch.ones(2, 4) * 2},
        ]
        predictor = _make_predictor(batches, per_batch)
        result = predictor.batch_collect_activations(layer_names=["layer"], dataset=predictor._dataset_mock)
        _, tensor = result[0]
        assert tensor.device.type == "cpu"

    def test_non_tensor_values_collected_as_lists(self):
        batches = [{"text_proc": np.zeros(2)} for _ in range(2)]
        per_batch = [
            {"used_tokens": 10},
            {"used_tokens": 20},
        ]
        predictor = _make_predictor(batches, per_batch)
        result = predictor.batch_collect_activations(layer_names=["used_tokens"], dataset=predictor._dataset_mock)
        name, values = result[0]
        assert name == "used_tokens"
        assert values == [10, 20]

    def test_bucketing_field_raises(self):
        from ludwig.models.predictor import Predictor

        predictor = Predictor.__new__(Predictor)
        predictor._distributed = MagicMock()
        predictor._distributed.rank.return_value = 0
        with pytest.raises(ValueError, match="BucketedBatcher"):
            predictor.batch_collect_activations(
                layer_names=["x"],
                dataset=MagicMock(),
                bucketing_field="some_field",
            )
