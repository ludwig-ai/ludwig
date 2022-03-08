#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
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
import pytest
from jsonschema.exceptions import ValidationError

from ludwig.constants import TRAINER
from ludwig.models.trainer import TrainerConfig
from ludwig.modules.optimization_modules import optimizer_registry
from ludwig.utils.schema import validate_config
from tests.integration_tests.utils import binary_feature, category_feature, number_feature

# Note: simple tests for now, but once we add dependent fields we can add tests for more complex relationships in this
# file. Currently verifies that the nested fields work, as the others are covered by basic marshmallow validation:


def test_config_trainer_empty_null_and_default():
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    validate_config(config)

    config[TRAINER] = None
    with pytest.raises(ValidationError):
        validate_config(config)

    config[TRAINER] = TrainerConfig.Schema().dump({})
    validate_config(config)


def test_config_trainer_bad_optimizer():
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    validate_config(config)

    # Test manually set-to-null optimizer vs unspecified:
    config[TRAINER]["optimizer"] = None
    with pytest.raises(ValidationError):
        validate_config(config)
    assert TrainerConfig.Schema().load({}).optimizer is not None

    # Test all types in optimizer_registry supported:
    for key in optimizer_registry.keys():
        config[TRAINER]["optimizer"] = {"type": key}
        validate_config(config)

    # Test invalid optimizer type:
    config[TRAINER]["optimizer"] = {"type": 0}
    with pytest.raises(ValidationError):
        validate_config(config)
    config[TRAINER]["optimizer"] = {"type": {}}
    with pytest.raises(ValidationError):
        validate_config(config)
    config[TRAINER]["optimizer"] = {"type": "invalid"}
    with pytest.raises(ValidationError):
        validate_config(config)


def test_optimizer_property_validation():
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    validate_config(config)

    # Test that an optimizer's property types are enforced:
    config[TRAINER]["optimizer"] = {"type": "rmsprop"}
    validate_config(config)

    config[TRAINER]["optimizer"]["momentum"] = "invalid"
    with pytest.raises(ValidationError):
        validate_config(config)

    # Test extra keys are excluded and defaults are loaded appropriately:
    config[TRAINER]["optimizer"]["momentum"] = 10
    config[TRAINER]["optimizer"]["extra_key"] = "invalid"
    validate_config(config)

    assert TrainerConfig.Schema().load(config[TRAINER]).optimizer.type == "rmsprop"
    assert TrainerConfig.Schema().load(config[TRAINER]).optimizer.momentum == 10
    assert TrainerConfig.Schema().load(config[TRAINER]).optimizer.eps == 1e-10
    assert not hasattr(TrainerConfig.Schema().load(config[TRAINER]).optimizer, "extra_key")

    # Test bad parameter range:
    config[TRAINER]["optimizer"] = {"type": "rmsprop", "eps": -1}
    with pytest.raises(ValidationError):
        validate_config(config)

    # Test config validation for tuple types:
    config[TRAINER]["optimizer"] = {"type": "adam", "betas": (0.1, 0.1)}
    validate_config(config)


def test_clipper_property_validation():
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {
            "type": "tabnet",
        },
        TRAINER: {},
    }
    validate_config(config)

    # Test null/empty clipper:
    assert TrainerConfig.Schema().load({}).gradient_clipping is not None

    config[TRAINER]["gradient_clipping"] = None
    validate_config(config)
    assert TrainerConfig.Schema().load(config[TRAINER]).gradient_clipping is None
    config[TRAINER]["gradient_clipping"] = {}
    validate_config(config)
    assert (
        TrainerConfig.Schema().load(config[TRAINER]).gradient_clipping
        == TrainerConfig.Schema().load({}).gradient_clipping
    )

    # Test invalid clipper type:
    config[TRAINER]["gradient_clipping"] = 0
    with pytest.raises(ValidationError):
        validate_config(config)
    config[TRAINER]["gradient_clipping"] = "invalid"
    with pytest.raises(ValidationError):
        validate_config(config)

    # Test that an optimizer's property types are enforced:
    config[TRAINER]["gradient_clipping"] = {"clipglobalnorm": None}
    validate_config(config)
    config[TRAINER]["gradient_clipping"] = {"clipglobalnorm": 1}
    validate_config(config)
    config[TRAINER]["gradient_clipping"] = {"clipglobalnorm": "invalid"}
    with pytest.raises(ValidationError):
        validate_config(config)

    # Test extra keys are excluded and defaults are loaded appropriately:
    config[TRAINER]["gradient_clipping"] = {"clipnorm": 1}
    config[TRAINER]["gradient_clipping"]["extra_key"] = "invalid"
    validate_config(config)

    assert TrainerConfig.Schema().load(config[TRAINER]).gradient_clipping.clipnorm == 1
    assert TrainerConfig.Schema().load(config[TRAINER]).gradient_clipping.clipglobalnorm == 0.5
    assert not hasattr(TrainerConfig.Schema().load(config[TRAINER]).gradient_clipping, "extra_key")
