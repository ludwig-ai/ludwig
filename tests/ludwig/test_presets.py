"""Tests for quality presets."""

import pytest

from ludwig.presets import _deep_merge, apply_preset, QUALITY_PRESETS


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}, "y": 3}
        override = {"x": {"b": 99, "c": 100}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 100}, "y": 3}

    def test_override_wins_for_non_dict(self):
        base = {"x": {"a": 1}}
        override = {"x": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"x": "replaced"}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        override = {"x": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"x": {"a": 1}}
        assert override == {"x": {"b": 2}}


class TestApplyPreset:
    def test_medium_quality_applies(self):
        config = {
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "number"}],
        }
        result = apply_preset(config, "medium_quality")
        assert result["combiner"]["type"] == "concat"
        assert result["trainer"]["epochs"] == 50
        # User features preserved
        assert result["input_features"] == config["input_features"]

    def test_high_quality_applies(self):
        config = {
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "number"}],
        }
        result = apply_preset(config, "high_quality")
        assert result["combiner"]["type"] == "transformer"
        assert result["trainer"]["loss_balancing"] == "uncertainty"

    def test_best_quality_applies(self):
        config = {
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "number"}],
        }
        result = apply_preset(config, "best_quality")
        assert result["combiner"]["type"] == "ft_transformer"
        assert result["trainer"]["model_soup"] == "uniform"

    def test_user_override_wins(self):
        config = {
            "input_features": [{"name": "x", "type": "number"}],
            "output_features": [{"name": "y", "type": "number"}],
            "combiner": {"type": "tabnet"},  # User override
            "trainer": {"epochs": 10},  # User override
        }
        result = apply_preset(config, "best_quality")
        # User overrides should win
        assert result["combiner"]["type"] == "tabnet"
        assert result["trainer"]["epochs"] == 10
        # But preset values not overridden should still apply
        assert result["trainer"]["model_soup"] == "uniform"

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            apply_preset({}, "nonexistent_preset")

    def test_all_presets_exist(self):
        assert "medium_quality" in QUALITY_PRESETS
        assert "high_quality" in QUALITY_PRESETS
        assert "best_quality" in QUALITY_PRESETS
