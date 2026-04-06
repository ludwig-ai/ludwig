"""Tests for model card generation."""

import os
import tempfile

from ludwig.utils.model_card import generate_model_card, save_model_card


class TestGenerateModelCard:
    def _make_config(self):
        return {
            "model_type": "ecd",
            "input_features": [
                {"name": "age", "type": "number"},
                {"name": "workclass", "type": "category"},
            ],
            "output_features": [{"name": "income", "type": "binary"}],
            "combiner": {"type": "ft_transformer", "hidden_size": 192, "num_heads": 8},
            "trainer": {"learning_rate": 0.001, "epochs": 100, "batch_size": 256},
        }

    def _make_metadata(self):
        return {
            "age": {"mean": 38.5, "std": 13.2, "preprocessing": {"normalization": "zscore"}},
            "workclass": {"idx2str": ["Private", "Self-emp", "Gov"]},
            "income": {"idx2str": ["<=50K", ">50K"]},
        }

    def test_basic_card(self):
        card = generate_model_card(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
        )
        assert isinstance(card, str)
        assert "# Model Card" in card
        assert "ECD" in card

    def test_contains_features(self):
        card = generate_model_card(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
        )
        assert "age" in card
        assert "workclass" in card
        assert "income" in card

    def test_contains_combiner(self):
        card = generate_model_card(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
        )
        assert "ft_transformer" in card

    def test_contains_trainer_info(self):
        card = generate_model_card(
            config=self._make_config(),
            training_set_metadata=self._make_metadata(),
        )
        assert "0.001" in card or "learning_rate" in card


class TestSaveModelCard:
    def test_save_creates_file(self):
        config = {
            "model_type": "ecd",
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "number"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model_card(
                output_directory=tmpdir,
                config=config,
                training_set_metadata={"x": {}, "y": {}},
            )
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "# Model Card" in content
