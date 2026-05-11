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
"""Unit tests for ludwig.automl.config_enumerator."""

from ludwig.automl.config_enumerator import (
    ConfigSpec,
    enumerate_config_specs,
    FeatureSpec,
    get_valid_combiners,
    get_valid_decoders,
    get_valid_encoders,
)

# ---------------------------------------------------------------------------
# get_valid_encoders
# ---------------------------------------------------------------------------


def test_get_valid_encoders_binary():
    encoders = get_valid_encoders("binary")
    assert set(encoders) == {"passthrough", "dense"}


def test_get_valid_encoders_text():
    encoders = get_valid_encoders("text")
    assert len(encoders) >= 10
    assert "bert" in encoders
    assert "distilbert" in encoders


def test_get_valid_encoders_unknown():
    encoders = get_valid_encoders("anomaly_type_that_does_not_exist")
    assert encoders == []


# ---------------------------------------------------------------------------
# get_valid_decoders
# ---------------------------------------------------------------------------


def test_get_valid_decoders_binary():
    decoders = get_valid_decoders("binary")
    assert set(decoders) == {"mlp_classifier", "regressor"}


def test_get_valid_decoders_number():
    decoders = get_valid_decoders("number")
    assert decoders == ["regressor"]


# ---------------------------------------------------------------------------
# get_valid_combiners
# ---------------------------------------------------------------------------


def test_get_valid_combiners_all_tabular():
    features = [
        FeatureSpec(name="a", type="binary"),
        FeatureSpec(name="b", type="category"),
        FeatureSpec(name="c", type="number"),
    ]
    combiners = get_valid_combiners(features)
    assert "tabnet" in combiners


def test_get_valid_combiners_with_text():
    features = [
        FeatureSpec(name="a", type="binary"),
        FeatureSpec(name="b", type="text"),
    ]
    combiners = get_valid_combiners(features)
    # tabnet requires all-tabular; text is not tabular
    assert "tabnet" not in combiners


def test_get_valid_combiners_comparator():
    # Exactly 2 inputs → comparator allowed
    two_features = [
        FeatureSpec(name="a", type="binary"),
        FeatureSpec(name="b", type="category"),
    ]
    combiners_two = get_valid_combiners(two_features)
    assert "comparator" in combiners_two

    # 3 inputs → comparator NOT allowed
    three_features = [
        FeatureSpec(name="a", type="binary"),
        FeatureSpec(name="b", type="category"),
        FeatureSpec(name="c", type="number"),
    ]
    combiners_three = get_valid_combiners(three_features)
    assert "comparator" not in combiners_three


def test_get_valid_combiners_sequence():
    # text input → sequence combiner allowed
    text_features = [
        FeatureSpec(name="a", type="text"),
        FeatureSpec(name="b", type="binary"),
    ]
    combiners = get_valid_combiners(text_features)
    assert "sequence" in combiners

    # binary-only → sequence combiner NOT allowed
    binary_only = [
        FeatureSpec(name="a", type="binary"),
        FeatureSpec(name="b", type="number"),
    ]
    combiners_no_seq = get_valid_combiners(binary_only)
    assert "sequence" not in combiners_no_seq


# ---------------------------------------------------------------------------
# enumerate_config_specs
# ---------------------------------------------------------------------------


def test_enumerate_config_specs_basic():
    inputs = [FeatureSpec(name="x", type="binary")]
    output = FeatureSpec(name="y", type="binary")
    specs = enumerate_config_specs(inputs, output)
    assert len(specs) > 0


def test_enumerate_config_specs_max_configs():
    inputs = [FeatureSpec(name="x", type="category"), FeatureSpec(name="z", type="number")]
    output = FeatureSpec(name="y", type="binary")
    specs = enumerate_config_specs(inputs, output, max_configs=5)
    assert len(specs) == 5


def test_enumerate_config_specs_no_valid_decoder():
    inputs = [FeatureSpec(name="x", type="binary")]
    # "anomaly" has no decoder registry entry
    output = FeatureSpec(name="y", type="anomaly")
    specs = enumerate_config_specs(inputs, output)
    assert specs == []


def test_config_spec_has_correct_fields():
    inputs = [FeatureSpec(name="x", type="binary")]
    output = FeatureSpec(name="y", type="binary")
    specs = enumerate_config_specs(inputs, output, max_configs=1)
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ConfigSpec)
    assert hasattr(spec, "input_encoders")
    assert hasattr(spec, "combiner")
    assert hasattr(spec, "output_decoder")
    assert hasattr(spec, "output_type")
    assert isinstance(spec.input_encoders, dict)
    assert "x" in spec.input_encoders
