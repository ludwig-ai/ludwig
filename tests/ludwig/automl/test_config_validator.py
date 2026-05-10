# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for ludwig.automl.config_validator."""

import numpy as np
import pandas as pd

from ludwig.automl.config_validator import validate_config_for_dataset, ValidationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feature_a": rng.integers(0, 10, size=n).astype(float),
            "feature_b": rng.standard_normal(n),
            "label": rng.choice([0, 1], size=n),
        }
    )


def _simple_config(batch_size: int | None = None) -> dict:
    cfg = {
        "input_features": [
            {"name": "feature_a", "type": "number"},
            {"name": "feature_b", "type": "number"},
        ],
        "output_features": [{"name": "label", "type": "binary"}],
        "combiner": {"type": "concat"},
    }
    if batch_size is not None:
        cfg["trainer"] = {"batch_size": batch_size}
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_config_passes():
    df = _make_binary_df(100)
    config = _simple_config()
    result = validate_config_for_dataset(config, df)
    assert result.is_valid


def test_missing_input_column():
    df = _make_binary_df(100)
    config = {
        "input_features": [{"name": "does_not_exist", "type": "number"}],
        "output_features": [{"name": "label", "type": "binary"}],
        "combiner": {"type": "concat"},
    }
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert len(result.failures) > 0


def test_missing_output_column():
    df = _make_binary_df(100)
    config = {
        "input_features": [{"name": "feature_a", "type": "number"}],
        "output_features": [{"name": "missing_label", "type": "binary"}],
        "combiner": {"type": "concat"},
    }
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert len(result.failures) > 0


def test_invalid_encoder_type():
    df = _make_binary_df(100)
    config = {
        "input_features": [
            {"name": "feature_a", "type": "binary", "encoder": {"type": "nonexistent_enc"}},
        ],
        "output_features": [{"name": "label", "type": "binary"}],
        "combiner": {"type": "concat"},
    }
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert any("nonexistent_enc" in f for f in result.failures)


def test_invalid_decoder_type():
    df = _make_binary_df(100)
    config = {
        "input_features": [{"name": "feature_a", "type": "number"}],
        "output_features": [{"name": "label", "type": "binary", "decoder": {"type": "nonexistent_dec"}}],
        "combiner": {"type": "concat"},
    }
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert any("nonexistent_dec" in f for f in result.failures)


def test_invalid_combiner_for_schema():
    # tabnet is only valid for all-tabular inputs; text is not tabular
    df = pd.DataFrame(
        {
            "sentence": ["hello world"] * 100,
            "label": [0, 1] * 50,
        }
    )
    config = {
        "input_features": [{"name": "sentence", "type": "text"}],
        "output_features": [{"name": "label", "type": "binary"}],
        "combiner": {"type": "tabnet"},
    }
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert any("tabnet" in f for f in result.failures)


def test_batch_size_too_large():
    df = _make_binary_df(50)
    config = _simple_config(batch_size=10000)
    result = validate_config_for_dataset(config, df)
    # batch_size >> dataset size → warning
    assert len(result.warnings) > 0


def test_output_in_input():
    df = _make_binary_df(100)
    config = {
        "input_features": [
            {"name": "feature_a", "type": "number"},
            {"name": "label", "type": "binary"},  # label is also an output
        ],
        "output_features": [{"name": "label", "type": "binary"}],
        "combiner": {"type": "concat"},
    }
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert any("label" in f for f in result.failures)


def test_single_class_output():
    df = _make_binary_df(100)
    df["label"] = 0  # all same value
    config = _simple_config()
    result = validate_config_for_dataset(config, df)
    assert not result.is_valid
    assert any("distinct" in f.lower() or "1" in f for f in result.failures)


def test_strict_mode():
    # A config with no combiner type generates a warning; strict=True should fail, strict=False pass
    df = _make_binary_df(100)
    config = {
        "input_features": [{"name": "feature_a", "type": "number"}],
        "output_features": [{"name": "label", "type": "binary"}],
        # combiner intentionally omitted → warning about missing combiner type
    }
    result_strict = validate_config_for_dataset(config, df, strict=True)
    result_lenient = validate_config_for_dataset(config, df, strict=False)

    # strict mode: warnings become failures
    assert not result_strict.is_valid
    # lenient mode: only hard failures matter
    assert result_lenient.is_valid


def test_validation_result_bool():
    assert bool(ValidationResult(is_valid=True)) is True
    assert bool(ValidationResult(is_valid=False)) is False
