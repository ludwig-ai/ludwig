"""Tests for model export utilities."""

import os
import tempfile

import torch
import torch.nn as nn

from ludwig.utils.model_export import load_exported_model, ModelExporter


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestModelExporter:
    def test_export_safetensors(self):
        model = SimpleModel()
        exporter = ModelExporter(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_safetensors(tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".safetensors")

    def test_export_torch(self):
        model = SimpleModel()
        exporter = ModelExporter(model)
        sample = torch.randn(2, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_torch(tmpdir, sample)
            assert os.path.exists(path)

    def test_generate_sample_input_fallback(self):
        model = SimpleModel()
        exporter = ModelExporter(model)
        sample = exporter._generate_sample_input()
        assert "input" in sample


class TestLoadExportedModel:
    def test_load_torchscript(self):
        model = SimpleModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            traced = torch.jit.trace(model, torch.randn(2, 10))
            traced.save(path)
            loaded = load_exported_model(path)
            assert loaded is not None

    def test_unknown_format_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown model format"):
            load_exported_model("model.xyz")
