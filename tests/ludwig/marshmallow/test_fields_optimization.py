#! /usr/bin/env python
from typing import Optional

import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import dataclass

import ludwig.marshmallow.marshmallow_schema_utils as lusutils
import ludwig.modules.optimization_modules as lmo

# Tests for custom dataclass/marshmallow fields:


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


def test_torch_description_pull():
    example_empty_desc_prop = lusutils.unload_jsonschema_from_marshmallow_class(lmo.AdamOptimizerConfig)["properties"][
        "lr"
    ]
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
    assert get_marshmallow_from_dataclass_field(default_optimizer_field).allow_none is False
    assert default_optimizer_field.default_factory() == lmo.AdamOptimizerConfig()

    # Test normal cases:
    optimizer_field = lmo.OptimizerDataclassField({"type": "adamax"})
    assert optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(optimizer_field).allow_none is False
    assert optimizer_field.default_factory() == lmo.AdamaxOptimizerConfig()

    optimizer_field = lmo.OptimizerDataclassField({"type": "adamax", "betas": (0.1, 0.1)})
    assert optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(optimizer_field).allow_none is False
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
        foo: Optional[lmo.BaseOptimizerConfig] = lmo.OptimizerDataclassField()

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lmo.AdamOptimizerConfig()

    # Test creating a schema with set default:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.BaseOptimizerConfig] = lmo.OptimizerDataclassField({"type": "adamax", "betas": (0.1, 0.1)})

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": None})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"type": "invalid", "betas": (0.2, 0.2)}})

    assert CustomTestSchema.Schema().load({}).foo == lmo.AdamaxOptimizerConfig(betas=(0.1, 0.1))
    assert CustomTestSchema.Schema().load(
        {"foo": {"type": "adamax", "betas": (0.2, 0.2)}}
    ).foo == lmo.AdamaxOptimizerConfig(betas=(0.2, 0.2))
    assert CustomTestSchema.Schema().load(
        {"foo": {"type": "adamax", "betas": (0.2, 0.2), "extra_key": 1}}
    ).foo == lmo.AdamaxOptimizerConfig(betas=(0.2, 0.2))


def test_ClipperDataclassField():
    # Test default case:
    default_clipper_field = lmo.GradientClippingDataclassField()
    assert default_clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(default_clipper_field).allow_none is True
    assert default_clipper_field.default_factory() == lmo.GradientClippingConfig()

    # Test normal cases:
    clipper_field = lmo.GradientClippingDataclassField({"clipglobalnorm": 0.1})
    assert clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(clipper_field).allow_none is True
    assert clipper_field.default_factory() == lmo.GradientClippingConfig(clipglobalnorm=0.1)

    clipper_field = lmo.GradientClippingDataclassField({"clipglobalnorm": None})
    assert clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(clipper_field).allow_none is True
    assert clipper_field.default_factory() == lmo.GradientClippingConfig(clipglobalnorm=None)

    # Test invalid default case:
    with pytest.raises(MarshmallowValidationError):
        lmo.GradientClippingDataclassField("test")
    with pytest.raises(MarshmallowValidationError):
        lmo.GradientClippingDataclassField(None)
    with pytest.raises(MarshmallowValidationError):
        lmo.GradientClippingDataclassField(1)

    # Test creating a schema with default options:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.GradientClippingConfig] = lmo.GradientClippingConfig()

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lmo.GradientClippingConfig()

    # Test creating a schema with set default:
    @dataclass
    class CustomTestSchema(lusutils.BaseMarshmallowConfig):
        foo: Optional[lmo.GradientClippingConfig] = lmo.GradientClippingDataclassField({"clipglobalnorm": 0.1})

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": "invalid"}})

    assert CustomTestSchema.Schema().load({}).foo == lmo.GradientClippingConfig(clipglobalnorm=0.1)
    assert CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": 1}}).foo == lmo.GradientClippingConfig(
        clipglobalnorm=1
    )
    assert CustomTestSchema.Schema().load(
        {"foo": {"clipglobalnorm": 1, "extra_key": 1}}
    ).foo == lmo.GradientClippingConfig(clipglobalnorm=1)
