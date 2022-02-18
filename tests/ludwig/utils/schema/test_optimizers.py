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
from typing import Optional

import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import dataclass
from marshmallow_jsonschema import JSONSchema as js

import ludwig.modules.optimization_modules as lmo
import ludwig.utils.schema_utils as lusutils

# Tests for utility methods:


def test_create_optimizer():
    pass


def test_create_optimizer_with_clipper():
    pass


# Tests for custom dataclass/marshmallow fields:


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


def test_torch_description_pull():
    example_empty_desc_prop = lusutils.get_custom_schema_from_marshmallow_class(lmo.AdamOptimizer)["properties"]["lr"]
    assert (
        isinstance(example_empty_desc_prop, dict)
        and "description" in example_empty_desc_prop
        and isinstance(example_empty_desc_prop["description"], str)
        and len(example_empty_desc_prop["description"]) > 3
    )


def test_OptimizerDataclassField():
    # Test default case:
    default_optimizer_field = lmo.OptimizerDataclassField()
    assert default_optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(default_optimizer_field).allow_none is True
    assert default_optimizer_field.default_factory() == lmo.AdamOptimizer()

    # Test normal cases:
    optimizer_field = lmo.OptimizerDataclassField({"type": "adamax"})
    assert optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(optimizer_field).allow_none is True
    assert optimizer_field.default_factory() == lmo.AdamaxOptimizer()

    optimizer_field = lmo.OptimizerDataclassField({"type": "adamax", "betas": (0.1, 0.1)})
    assert optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(optimizer_field).allow_none is True
    assert optimizer_field.default_factory().betas == (0.1, 0.1)

    # Test invalid default case:
    with pytest.raises(MarshmallowValidationError):
        lmo.OptimizerDataclassField({})
    with pytest.raises(MarshmallowValidationError):
        lmo.OptimizerDataclassField("test")
    with pytest.raises(MarshmallowValidationError):
        lmo.OptimizerDataclassField(None)
    with pytest.raises(MarshmallowValidationError):
        lmo.OptimizerDataclassField(1)

    # Test creating a schema with default options:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.BaseOptimizer] = lmo.OptimizerDataclassField()

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lmo.AdamOptimizer()

    # Test creating a schema with set default:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.BaseOptimizer] = lmo.OptimizerDataclassField({"type": "adamax", "betas": (0.1, 0.1)})

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"type": "invalid", "betas": (0.2, 0.2)}})

    assert CustomTestSchema.Schema().load({}).foo == lmo.AdamaxOptimizer(betas=(0.1, 0.1))
    assert CustomTestSchema.Schema().load({"foo": {"type": "adamax", "betas": (0.2, 0.2)}}).foo == lmo.AdamaxOptimizer(
        betas=(0.2, 0.2)
    )
    assert CustomTestSchema.Schema().load(
        {"foo": {"type": "adamax", "betas": (0.2, 0.2), "extra_key": 1}}
    ).foo == lmo.AdamaxOptimizer(betas=(0.2, 0.2))


def test_ClipperDataclassField():
    # Test default case:
    default_clipper_field = lmo.ClipperDataclassField()
    assert default_clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(default_clipper_field).allow_none is True
    assert default_clipper_field.default_factory() == lmo.Clipper()

    # Test normal cases:
    clipper_field = lmo.ClipperDataclassField({"clipglobalnorm": 0.1})
    assert clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(clipper_field).allow_none is True
    assert clipper_field.default_factory() == lmo.Clipper(clipglobalnorm=0.1)

    clipper_field = lmo.ClipperDataclassField({"clipglobalnorm": None})
    assert clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(clipper_field).allow_none is True
    assert clipper_field.default_factory() == lmo.Clipper(clipglobalnorm=None)

    # Test invalid default case:
    with pytest.raises(MarshmallowValidationError):
        lmo.ClipperDataclassField("test")
    with pytest.raises(MarshmallowValidationError):
        lmo.ClipperDataclassField(None)
    with pytest.raises(MarshmallowValidationError):
        lmo.ClipperDataclassField(1)

    # Test creating a schema with default options:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.Clipper] = lmo.Clipper()

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lmo.Clipper()

    # Test creating a schema with set default:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.Clipper] = lmo.ClipperDataclassField({"clipglobalnorm": 0.1})

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": "invalid"}})

    assert CustomTestSchema.Schema().load({}).foo == lmo.Clipper(clipglobalnorm=0.1)
    assert CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": 1}}).foo == lmo.Clipper(clipglobalnorm=1)
    assert CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": 1, "extra_key": 1}}).foo == lmo.Clipper(
        clipglobalnorm=1
    )

    # Test expected schema dumps:
    raw_clipper_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {
            "Clipper": {
                "type": "object",
                "properties": {
                    "clipglobalnorm": {
                        "title": "clipglobalnorm",
                        "type": ["number", "null"],
                        "format": "float",
                        "default": 0.5,
                    },
                    "clipnorm": {"title": "clipnorm", "type": ["number", "null"], "format": "float", "default": None},
                    "clipvalue": {"title": "clipvalue", "type": ["number", "null"], "format": "float", "default": None},
                },
                "additionalProperties": False,
            }
        },
        "$ref": "#/definitions/Clipper",
    }

    assert js().dump(lmo.Clipper.Schema()) == raw_clipper_schema

    raw_clipper_dataclass_schema = {
        "oneOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {
                    "clipglobalnorm": {
                        "title": "clipglobalnorm",
                        "type": ["number", "null"],
                        "format": "float",
                        "default": 0.5,
                        "description": "(default: 0. 5).",
                    },
                    "clipnorm": {
                        "title": "clipnorm",
                        "type": ["number", "null"],
                        "format": "float",
                        "default": None,
                        "description": "(default: None).",
                    },
                    "clipvalue": {
                        "title": "clipvalue",
                        "type": ["number", "null"],
                        "format": "float",
                        "default": None,
                        "description": "(default: None).",
                    },
                },
                "additionalProperties": True,
                "description": "Dataclass that holds gradient clipping parameters.",
            },
        ]
    }

    assert lmo.ClipperMarshmallowField()._jsonschema_type_mapping() == raw_clipper_dataclass_schema
