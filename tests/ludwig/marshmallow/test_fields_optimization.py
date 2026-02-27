#! /usr/bin/env python

import pytest
from pydantic import ValidationError as PydanticValidationError

import ludwig.schema.optimizers as lso
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


def test_torch_description_pull():
    example_empty_desc_prop = schema_utils.unload_jsonschema_from_marshmallow_class(lso.AdamOptimizerConfig)[
        "properties"
    ]["eps"]
    assert (
        isinstance(example_empty_desc_prop, dict)
        and "description" in example_empty_desc_prop
        and isinstance(example_empty_desc_prop["description"], str)
        and len(example_empty_desc_prop["description"]) > 3
    )


def test_OptimizerDataclassField():
    # Test default case:
    default_optimizer_field = lso.OptimizerDataclassField()
    assert default_optimizer_field.default_factory is not None
    assert default_optimizer_field.default_factory() == lso.AdamOptimizerConfig()

    # Test normal cases:
    optimizer_field = lso.OptimizerDataclassField("adamax")
    assert optimizer_field.default_factory is not None
    assert optimizer_field.default_factory() == lso.AdamaxOptimizerConfig()

    # Test invalid default case:
    with pytest.raises(AttributeError):
        lso.OptimizerDataclassField({})
    with pytest.raises(KeyError):
        lso.OptimizerDataclassField("test")
    with pytest.raises(AttributeError):
        lso.OptimizerDataclassField(1)

    # Test creating a schema with default options:
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: lso.BaseOptimizerConfig | None = lso.OptimizerDataclassField()

    with pytest.raises((PydanticValidationError, Exception)):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lso.AdamOptimizerConfig()

    # Test creating a schema with set default:
    @ludwig_dataclass
    class CustomTestSchema2(schema_utils.BaseMarshmallowConfig):
        foo: lso.BaseOptimizerConfig | None = lso.OptimizerDataclassField("adamax")

    with pytest.raises((PydanticValidationError, Exception)):
        CustomTestSchema2.Schema().load({"foo": "test"})

    assert CustomTestSchema2.Schema().load(
        {"foo": {"type": "adamax", "betas": (0.2, 0.2)}}
    ).foo == lso.AdamaxOptimizerConfig(betas=(0.2, 0.2))


def test_ClipperDataclassField():
    # Test default case:
    default_clipper_field = lso.GradientClippingDataclassField(description="", default={})
    assert default_clipper_field.default_factory is not None
    assert default_clipper_field.default_factory() == lso.GradientClippingConfig()

    # Test normal cases:
    clipper_field = lso.GradientClippingDataclassField(description="", default={"clipglobalnorm": 0.1})
    assert clipper_field.default_factory is not None
    assert clipper_field.default_factory() == lso.GradientClippingConfig(clipglobalnorm=0.1)

    clipper_field = lso.GradientClippingDataclassField(description="", default={"clipglobalnorm": None})
    assert clipper_field.default_factory is not None
    assert clipper_field.default_factory() == lso.GradientClippingConfig(clipglobalnorm=None)

    # Test invalid default case:
    with pytest.raises(Exception):
        lso.GradientClippingDataclassField(description="", default="test")
    with pytest.raises(Exception):
        lso.GradientClippingDataclassField(description="", default=None)
    with pytest.raises(Exception):
        lso.GradientClippingDataclassField(description="", default=1)

    # Test creating a schema with set default:
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: lso.GradientClippingConfig | None = lso.GradientClippingDataclassField(
            description="", default={"clipglobalnorm": 0.1}
        )

    with pytest.raises((PydanticValidationError, Exception)):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lso.GradientClippingConfig(clipglobalnorm=0.1)
    assert CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": 1}}).foo == lso.GradientClippingConfig(
        clipglobalnorm=1
    )
