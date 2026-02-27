import pytest
from pydantic import ValidationError as PydanticValidationError

from ludwig.config_validation.validation import get_validator, validate
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


def get_marshmallow_field_from_metadata(dfield):
    """Helper method for extracting the marshmallow field from pydantic field metadata."""
    metadata = dfield.metadata
    if isinstance(metadata, dict):
        return metadata.get("marshmallow_field")
    if isinstance(metadata, (list, tuple)):
        for item in metadata:
            if hasattr(item, "_deserialize"):
                return item
    return None


# Simple marshmallow fields:


def test_StringOptions():
    # Test case of default conflicting with allowed options:
    test_options = ["one"]
    with pytest.raises(AssertionError):
        schema_utils.StringOptions(test_options, default=None, allow_none=False)

    # Test creating a schema with simple option, null not allowed:
    test_options = ["one"]

    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: str = schema_utils.StringOptions(test_options, "one", allow_none=False)

    with pytest.raises(PydanticValidationError):
        CustomTestSchema.Schema().load({"foo": None})


# Complex, custom marshmallow fields:


def test_Embed():
    # Test simple schema creation:
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: str | int | None = schema_utils.Embed()

    # Test null/empty loading cases:
    assert CustomTestSchema.Schema().load({}).foo is None
    assert CustomTestSchema.Schema().load({"foo": None}).foo is None

    # Test valid strings/numbers:
    assert CustomTestSchema.Schema().load({"foo": "add"}).foo == "add"
    assert CustomTestSchema.Schema().load({"foo": 1}).foo == 1


def test_InitializerOrDict():
    # Test default value validation:
    with pytest.raises(Exception):
        schema_utils.InitializerOrDict("test")

    # Test simple schema creation:
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: str | dict | None = schema_utils.InitializerOrDict()

    # Test valid loads:
    assert CustomTestSchema.Schema().load({}).foo == "xavier_uniform"
    assert CustomTestSchema.Schema().load({"foo": "zeros"}).foo == "zeros"

    # Test valid dict loads:
    assert CustomTestSchema.Schema().load({"foo": {"type": "zeros"}}).foo == {"type": "zeros"}


def test_FloatRangeTupleDataclassField():
    # Test dimensional mismatch:
    with pytest.raises(Exception):
        schema_utils.FloatRangeTupleDataclassField(n=3, default=(1, 1))

    # Test default schema creation:
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: tuple[float, float] | None = schema_utils.FloatRangeTupleDataclassField(allow_none=True)

    # Test empty load:
    assert CustomTestSchema.Schema().load({}).foo == (0.9, 0.999)
    assert CustomTestSchema.Schema().load({"foo": None}).foo is None

    # Test valid loads:
    assert CustomTestSchema.Schema().load({"foo": [0.5, 0.6]}).foo == (0.5, 0.6)

    # Test non-default schema (N=3, other custom metadata):
    @ludwig_dataclass
    class CustomTestSchema2(schema_utils.BaseMarshmallowConfig):
        foo: tuple[float, float, float] | None = schema_utils.FloatRangeTupleDataclassField(
            n=3, default=(1, 1, 1), min=-10, max=10
        )

    assert CustomTestSchema2.Schema().load({}).foo == (1, 1, 1)
    assert CustomTestSchema2.Schema().load({"foo": [2, 2, 2]}).foo == (2, 2, 2)


def test_OneOfOptionsField():
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: float | str = schema_utils.OneOfOptionsField(
            default=0.1,
            description="",
            allow_none=False,
            field_options=[
                schema_utils.FloatRange(default=0.001, min=0, max=1, allow_none=False),
                schema_utils.StringOptions(options=["placeholder"], default="placeholder", allow_none=False),
            ],
        )

    # Test valid loads:
    assert CustomTestSchema.Schema().load({}).foo == 0.1
    assert CustomTestSchema().foo == 0.1

    # Reverse the order and allow none (via StringOptions):
    @ludwig_dataclass
    class CustomTestSchema2(schema_utils.BaseMarshmallowConfig):
        foo: float | str | None = schema_utils.OneOfOptionsField(
            default="placeholder",
            description="",
            field_options=[
                schema_utils.FloatRange(default=0.001, min=0, max=1, allow_none=False),
                schema_utils.StringOptions(options=["placeholder"], default="placeholder", allow_none=False),
            ],
            allow_none=True,
        )

    # Test valid loads:
    assert CustomTestSchema2.Schema().load({}).foo == "placeholder"
    assert CustomTestSchema2.Schema().load({"foo": 0.1}).foo == 0.1
    assert CustomTestSchema2().foo == "placeholder"
    CustomTestSchema2.Schema().load({"foo": None})

    # Test JSON schema generation:
    json = schema_utils.unload_jsonschema_from_marshmallow_class(CustomTestSchema2)
    assert "foo" in json["properties"]


def test_OneOfOptionsField_allows_none():
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: float | str | None = schema_utils.OneOfOptionsField(
            default=None,
            allow_none=True,
            description="",
            field_options=[
                schema_utils.PositiveInteger(description="", default=1, allow_none=False),
                schema_utils.List(list_type=int, allow_none=False),
            ],
        )

    json = schema_utils.unload_jsonschema_from_marshmallow_class(CustomTestSchema)
    schema = {
        "type": "object",
        "properties": {
            "hello": json,
        },
        "definitions": {},
    }
    validate(instance={"hello": {"foo": None}}, schema=schema, cls=get_validator())


def test_OneOfOptionsField_multiple_fields_allow_none():
    # With pydantic, multiple fields allowing none is handled by union validation.
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: float | str | None = schema_utils.OneOfOptionsField(
            default=None,
            description="",
            field_options=[
                schema_utils.PositiveInteger(description="", default=1, allow_none=True),
                schema_utils.List(list_type=int, allow_none=True),
            ],
        )

    assert CustomTestSchema().foo is None


def test_OneOfOptionsField_allows_none_one_field_allows_none():
    @ludwig_dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: float | str | None = schema_utils.OneOfOptionsField(
            default=None,
            description="",
            field_options=[
                schema_utils.PositiveInteger(description="", default=1, allow_none=False),
                schema_utils.List(list_type=int, allow_none=True),
            ],
        )

    json = schema_utils.unload_jsonschema_from_marshmallow_class(CustomTestSchema)
    schema = {
        "type": "object",
        "properties": {
            "hello": json,
        },
        "definitions": {},
    }
    validate(instance={"hello": {"foo": None}}, schema=schema, cls=get_validator())
