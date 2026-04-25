"""Tests for training report generation."""

import json
import os
import tempfile
from types import SimpleNamespace

from ludwig.utils.training_report import generate_training_report, save_training_report


class TestGenerateTrainingReport:
    def _make_config(self):
        return {
            "model_type": "ecd",
            "input_features": [
                {"name": "age", "type": "number"},
                {"name": "workclass", "type": "category"},
            ],
            "output_features": [{"name": "income", "type": "binary"}],
            "combiner": {"type": "concat"},
        }

    def _make_metadata(self):
        return {
            "age": {"mean": 38.5, "std": 13.2},
            "workclass": {"idx2str": ["Private", "Self-emp", "Gov"]},
            "income": {"idx2str": ["<=50K", ">50K"]},
        }

    def test_basic_report(self):
        report = generate_training_report(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
        )
        assert report["report_version"] == "1.0"
        assert "generated_at" in report
        assert report["model_type"] == "ecd"
        assert "environment" in report
        assert "python_version" in report["environment"]

    def test_data_schema(self):
        report = generate_training_report(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
        )
        schema = report["data_schema"]
        assert len(schema["input_features"]) == 2
        assert len(schema["output_features"]) == 1
        # Number feature should have mean/std
        age_feat = schema["input_features"][0]
        assert age_feat["name"] == "age"
        assert age_feat["mean"] == 38.5
        # Category feature should have vocab_size
        wc_feat = schema["input_features"][1]
        assert wc_feat["vocab_size"] == 3

    def test_with_train_stats(self):
        train_stats = SimpleNamespace(
            training={"combined": {"loss": [0.5, 0.3, 0.2]}},
            validation={"combined": {"loss": [0.6, 0.4, 0.35]}},
            test=None,
        )
        report = generate_training_report(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
            train_stats=train_stats,
        )
        assert report["epochs_trained"] == 3
        assert "metrics" in report
        assert report["metrics"]["training"]["combined"]["loss"]["best"] == 0.2

    def test_with_timing(self):
        report = generate_training_report(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
            training_time_seconds=145.678,
        )
        assert report["training_time_seconds"] == 145.68

    def test_with_random_seed(self):
        report = generate_training_report(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
            random_seed=42,
        )
        assert report["random_seed"] == 42


class TestSaveTrainingReport:
    def test_save_creates_file(self):
        config = {
            "model_type": "ecd",
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "number"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_training_report(
                output_directory=tmpdir,
                config=config,
                training_set_metadata={"x": {}, "y": {}},
            )
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["model_type"] == "ecd"
