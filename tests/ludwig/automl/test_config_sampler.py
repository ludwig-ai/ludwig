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
"""Unit tests for ludwig.automl.config_sampler."""

from ludwig.automl.config_enumerator import FeatureSpec
from ludwig.automl.config_sampler import sample_configs
from ludwig.constants import COMBINER, INPUT_FEATURES, OUTPUT_FEATURES, TRAINER, TYPE

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TABULAR_INPUTS = [
    FeatureSpec(name="age", type="number"),
    FeatureSpec(name="income", type="number"),
    FeatureSpec(name="category_col", type="category"),
    FeatureSpec(name="flag", type="binary"),
]
_BINARY_OUTPUT = FeatureSpec(name="label", type="binary")

_TEXT_INPUTS = [FeatureSpec(name="sentence", type="text")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sample_configs_returns_n():
    results = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=10, seed=0)
    assert len(results) <= 10
    assert len(results) > 0


def test_sample_configs_deduplication():
    results = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=5, seed=7)
    hashes = [r.config_hash for r in results]
    assert len(hashes) == len(set(hashes)), "Duplicate config hashes found"


def test_sample_configs_combiner_diversity():
    results = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=50, seed=42)
    combiners_seen = {r.spec.combiner for r in results}
    assert len(combiners_seen) >= 3, f"Expected ≥3 distinct combiners, got {combiners_seen}"


def test_sample_configs_reproducible():
    results_a = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=10, seed=99)
    results_b = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=10, seed=99)
    hashes_a = [r.config_hash for r in results_a]
    hashes_b = [r.config_hash for r in results_b]
    assert hashes_a == hashes_b, "Same seed should produce identical configs"


def test_sample_configs_different_seeds():
    results_a = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=20, seed=1)
    results_b = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=20, seed=2)
    hashes_a = {r.config_hash for r in results_a}
    hashes_b = {r.config_hash for r in results_b}
    # They may share some configs, but should not be identical sets for a large sample
    assert hashes_a != hashes_b, "Different seeds should not produce identical config sets"


def test_sampled_config_has_valid_structure():
    results = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=3, seed=0)
    assert len(results) > 0
    cfg = results[0].config_dict
    assert INPUT_FEATURES in cfg
    assert OUTPUT_FEATURES in cfg
    assert COMBINER in cfg
    assert TRAINER in cfg


def test_sample_configs_text_schema():
    results = sample_configs(_TEXT_INPUTS, _BINARY_OUTPUT, n=5, seed=0)
    assert len(results) > 0


def test_sample_configs_empty_schema():
    # sample_configs with empty input features still generates configs (combiner/decoder axes are non-empty);
    # the result is a non-empty list with configs that have empty input_features.
    results = sample_configs([], _BINARY_OUTPUT, n=10, seed=0)
    # Each config should declare an empty input_features list
    for r in results:
        assert r.config_dict[INPUT_FEATURES] == []


def test_build_config_dict_structure():
    results = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=3, seed=0)
    assert len(results) > 0
    input_features = results[0].config_dict[INPUT_FEATURES]
    for feat_dict in input_features:
        assert "name" in feat_dict
        assert TYPE in feat_dict
        assert "encoder" in feat_dict


def test_combiner_params_in_config():
    # Force a tabnet config by running enough samples
    results = sample_configs(_TABULAR_INPUTS, _BINARY_OUTPUT, n=50, seed=42)
    tabnet_results = [r for r in results if r.spec.combiner == "tabnet"]
    assert len(tabnet_results) > 0, "Expected at least one tabnet config in 50 samples"
    combiner_dict = tabnet_results[0].config_dict[COMBINER]
    assert "size" in combiner_dict
    assert "num_steps" in combiner_dict
