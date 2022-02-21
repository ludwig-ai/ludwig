import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow.utils import EXCLUDE
from marshmallow_dataclass import dataclass
from marshmallow_jsonschema import JSONSchema as js

import ludwig.combiners.combiners as lcc
import ludwig.utils.schema_utils as lusutils
from ludwig.models.trainer import TrainerConfig


def test_get_fully_qualified_class_name():
    # Simple examples (marshmallow and non-marshmallow classes):
    assert lusutils.get_fully_qualified_class_name(TrainerConfig) == "ludwig.models.trainer.TrainerConfig"
    assert lusutils.get_fully_qualified_class_name(lcc.ConcatCombiner) == "ludwig.combiners.combiners.ConcatCombiner"


def test_assert_is_a_marshmallow_clas():
    lusutils.assert_is_a_marshmallow_class(TrainerConfig)
    with pytest.raises(AssertionError, match=r"^Expected marshmallow class.*"):
        lusutils.assert_is_a_marshmallow_class(lcc.ConcatCombiner)


# Note: testing class declared outside of function scope here so that name resolution works properly:
@dataclass
class CustomTestSchema(lusutils.BaseMarshmallowConfig):
    """sample docstring."""

    foo: int = 5
    "foo (default: 5)"


def test_custom_marshmallow_inheritance():
    assert CustomTestSchema.Meta.unknown == EXCLUDE


def test_unload_schema_from_marshmallow_jsonschema_dump():
    raw_test_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {
            "CustomTestSchema": {
                "properties": {"foo": {"title": "foo", "type": "integer", "default": 5}},
                "type": "object",
                "additionalProperties": False,
            }
        },
        "$ref": "#/definitions/CustomTestSchema",
    }

    unloaded_test_schema = {
        "properties": {"foo": {"title": "foo", "type": "integer", "default": 5}},
        "type": "object",
        "additionalProperties": False,
    }

    assert js().dump(CustomTestSchema.Schema()) == raw_test_schema
    assert lusutils.unload_schema_from_marshmallow_jsonschema_dump(CustomTestSchema) == unloaded_test_schema


def test_get_custom_schema_from_marshmallow_class():
    custom_test_schema = {
        "properties": {"foo": {"title": "foo", "type": "integer", "default": 5, "description": "Foo (default: 5)."}},
        "type": "object",
        "additionalProperties": True,
        "description": "Sample docstring.",
    }

    assert lusutils.get_custom_schema_from_marshmallow_class(CustomTestSchema) == custom_test_schema


def test_load_config_with_kwargs():
    test_kwargs = {
        "foo": 6,
        "bar": 6,
    }
    initialized_class, leftover = lusutils.load_config_with_kwargs(CustomTestSchema, test_kwargs)

    assert initialized_class.foo == 6
    assert leftover == {"bar": 6}

    # TransformerCombiner has no required/non-default arguments:
    initialized_class, leftover = lusutils.load_config_with_kwargs(lcc.TransformerCombinerConfig, test_kwargs)
    assert initialized_class.bias_initializer == "zeros"
    assert leftover == test_kwargs
    initialized_class, leftover = lusutils.load_config_with_kwargs(lcc.TransformerCombinerConfig, {})
    assert leftover == {}

    # ComparatorCombiner does have required arguments, so expect a failure:
    with pytest.raises(MarshmallowValidationError):
        initialized_class, leftover = lusutils.load_config_with_kwargs(lcc.ComparatorCombinerConfig, test_kwargs)
