"""Tests for ModelInspector."""

import torch
import torch.nn as nn

from ludwig.model_inspector import ModelInspector


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TestModelInspector:
    def test_collect_weights(self):
        model = SimpleModel()
        inspector = ModelInspector(model, {}, {})
        weights = inspector.collect_weights()
        assert len(weights) == 4  # 2 weight + 2 bias
        assert all("name" in w for w in weights)
        assert all("shape" in w for w in weights)

    def test_collect_specific_weights(self):
        model = SimpleModel()
        inspector = ModelInspector(model, {}, {})
        weights = inspector.collect_weights(tensor_names=["linear1.weight"])
        assert len(weights) == 1
        assert weights[0]["name"] == "linear1.weight"

    def test_model_summary(self):
        model = SimpleModel()
        config = {
            "model_type": "ecd",
            "combiner": {"type": "concat"},
            "input_features": [{"name": "x"}],
            "output_features": [{"name": "y"}],
        }
        inspector = ModelInspector(model, config, {})
        summary = inspector.model_summary()
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] == summary["total_parameters"]
        assert summary["model_size_mb"] >= 0
        assert summary["model_type"] == "ecd"
        assert "Linear" in summary["layer_counts"]
