from dataclasses import field
from typing import Any
from typing import Dict as TDict
from typing import List as TList
from typing import Tuple, Type, Union

from marshmallow import EXCLUDE, fields, schema, validate, ValidationError
from marshmallow_dataclass import dataclass as m_dataclass
from marshmallow_jsonschema import JSONSchema as js

from ludwig.modules.reduction_modules import reduce_mode_registry
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.utils.torch_utils import activations, initializer_registry


def load_config(cls: Type["BaseMarshmallowConfig"], **kwargs) -> "BaseMarshmallowConfig":  # noqa 0821
    """Takes a marshmallow class and instantiates it with the given keyword args as parameters."""
    assert_is_a_marshmallow_class(cls)
    schema = cls.Schema()
    return schema.load(kwargs)


def load_trainer_with_kwargs(
    model_type: str, kwargs: dict
) -> Tuple["BaseMarshmallowConfig", TDict[str, Any]]:  # noqa: F821
    """Special case of `load_config_with_kwargs` for the trainer schemas.

    In particular, it chooses the correct default type for an incoming config (if it doesn't have one already), but
    otherwise passes all other parameters through without change.
    """
    from ludwig.constants import MODEL_ECD, TYPE
    from ludwig.schema.trainer import ECDTrainerConfig, GBMTrainerConfig

    trainer_schema = ECDTrainerConfig if model_type == MODEL_ECD else GBMTrainerConfig

    def default_type_for_trainer_schema(cls):
        """Returns the default values for the "type" field on the given trainer schema."""
        return cls.Schema().fields[TYPE].dump_default

    # Create a copy of kwargs with the correct default type (which will be overridden if kwargs already contains 'type')
    kwargs_with_type = {**{TYPE: default_type_for_trainer_schema(trainer_schema)}, **kwargs}

    return load_config_with_kwargs(trainer_schema, kwargs_with_type)


def load_config_with_kwargs(
    cls: Type["BaseMarshmallowConfig"], kwargs_overrides
) -> Tuple["BaseMarshmallowConfig", TDict[str, Any]]:  # noqa 0821
    """Instatiates an instance of the marshmallow class and kwargs overrides instantiantes the schema.

    Returns a tuple of config, and a dictionary of any keys in kwargs_overrides which are no present in config.
    """
    assert_is_a_marshmallow_class(cls)
    schema = cls.Schema()
    fields = schema.fields.keys()
    return load_config(cls, **{k: v for k, v in kwargs_overrides.items() if k in fields}), {
        k: v for k, v in kwargs_overrides.items() if k not in fields
    }


def create_cond(if_pred: TDict, then_pred: TDict):
    """Returns a JSONSchema conditional for the given if-then predicates."""
    return {
        "if": {"properties": {k: {"const": v} for k, v in if_pred.items()}},
        "then": {"properties": {k: v for k, v in then_pred.items()}},
    }


class BaseMarshmallowConfig:
    """Base marshmallow class for common attributes and metadata."""

    class Meta:
        """Sub-class specifying meta information for Marshmallow.

        Currently only sets `unknown` flag to `EXCLUDE`. This is done to mirror Ludwig behavior: unknown properties are
        excluded from `load` calls so that the marshmallow_dataclass package can be used but
        `unload_jsonschema_from_marshmallow_class` will manually set a marshmallow schema's `additionalProperties` attr.
        to True so that JSON objects with extra properties do not raise errors; as a result properties are picked and
        filled in as necessary.
        """

        unknown = EXCLUDE
        "Flag that sets marshmallow `load` calls to ignore unknown properties passed as a parameter."


def assert_is_a_marshmallow_class(cls):
    assert hasattr(cls, "Schema") and isinstance(
        cls.Schema, schema.SchemaMeta
    ), f"Expected marshmallow class, but `{cls}` does not have the necessary `Schema` attribute."


def unload_jsonschema_from_marshmallow_class(mclass) -> TDict:
    """Helper method to directly get a marshmallow class's JSON schema without extra wrapping props."""
    assert_is_a_marshmallow_class(mclass)
    schema = js().dump(mclass.Schema())["definitions"][mclass.__name__]
    schema["additionalProperties"] = True
    return schema


def InitializerOptions(default: str = "xavier_uniform", description=""):
    """Utility wrapper that returns a `StringOptions` field with keys from `initializer_registry`."""
    return StringOptions(list(initializer_registry.keys()), default=default, allow_none=False, description=description)


def ActivationOptions(default: str = "relu", description=""):
    """Utility warapper that returns a `StringOptions` field with keys from `activations` registry."""
    return StringOptions(list(activations.keys()), default=default, allow_none=True, description=description)


def ReductionOptions(default: Union[None, str] = None, description=""):
    """Utility wrapper that returns a `StringOptions` field with keys from `reduce_mode_registry`."""
    return StringOptions(list(reduce_mode_registry.keys()), default=default, allow_none=True, description=description)


def RegularizerOptions(default: Union[None, str] = None, allow_none: bool = True, description=""):
    """Utility wrapper that returns a `StringOptions` field with prefilled regularizer options."""
    return StringOptions(["l1", "l2", "l1_l2"], default=default, allow_none=allow_none, description=description)


def String(
    description: str,
    default: Union[None, str] = None,
    allow_none: bool = True,
    pattern: str = None,
    parameter_metadata: ParameterMetadata = None,
):
    if not allow_none and not isinstance(default, str):
        raise ValidationError(f"Provided default `{default}` should be a string!")

    if pattern is not None:
        validation = validate.Regexp(pattern)
    else:
        validation = None

    return field(
        metadata={
            "marshmallow_field": fields.String(
                validate=validation,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={"description": description},
            ),
            "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
        },
        default=default,
    )


def StringOptions(
    options: TList[str],
    default: Union[None, str] = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata that enforces string inputs must be one of `options`.

    By default, None is allowed (and automatically appended) to the allowed list of options.
    """
    # If None should be allowed for an enum field, it also has to be defined as a valid
    # [option](https://github.com/json-schema-org/json-schema-spec/issues/258):
    if len(options) <= 0:
        raise ValidationError("Must provide non-empty list of options!")
    if default is not None and not isinstance(default, str):
        raise ValidationError(f"Provided default `{default}` should be a string!")
    if allow_none and None not in options:
        options += [None]
    if not allow_none and None in options:
        options.remove(None)
    if default not in options:
        raise ValidationError(f"Provided default `{default}` is not one of allowed options: {options} ")
    return field(
        metadata={
            "marshmallow_field": fields.String(
                validate=validate.OneOf(options),
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def Boolean(default: bool, description: str, parameter_metadata: ParameterMetadata = None):
    if default is not None:
        try:
            assert isinstance(default, bool)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Boolean(
                truthy={True},
                falsy={False},
                allow_none=False,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def Integer(
    default: Union[None, int] = None, allow_none=False, description="", parameter_metadata: ParameterMetadata = None
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs."""
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, int)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                strict=True,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def PositiveInteger(
    description: str, default: Union[None, int], allow_none: bool = False, parameter_metadata: ParameterMetadata = None
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs must be
    positive."""
    val = validate.Range(min=1)
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, int)
            val(default)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                strict=True,
                validate=val,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def NonNegativeInteger(
    description: str,
    default: Union[None, int] = None,
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs must be
    nonnegative."""
    val = validate.Range(min=0)
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, int)
            val(default)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                strict=True,
                validate=val,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def IntegerRange(
    description: str,
    default: Union[None, int] = None,
    allow_none=False,
    parameter_metadata: ParameterMetadata = None,
    min: int = None,
    max: int = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs must be in range
    set by relevant keyword args."""
    val = validate.Range(min=min, max=max, min_inclusive=min_inclusive, max_inclusive=max_inclusive)
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, int)
            val(default)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                strict=True,
                validate=val,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def NonNegativeFloat(
    default: Union[None, float] = None,
    allow_none=False,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata enforcing numeric inputs must be nonnegative."""
    val = validate.Range(min=0.0)
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, float) or isinstance(default, int)
            val(default)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Float(
                validate=val,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def FloatRange(
    default: Union[None, float] = None,
    allow_none: bool = False,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
    min: int = None,
    max: int = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    """Returns a dataclass field with marshmallow metadata enforcing numeric inputs must be in range set by
    relevant keyword args."""
    val = validate.Range(min=min, max=max, min_inclusive=min_inclusive, max_inclusive=max_inclusive)
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, float) or isinstance(default, int)
            val(default)
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Float(
                validate=val,
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


def IntegerOrSequenceOfIntegers(
    default: Union[None, int, Tuple[int, ...], TList[int]] = None,
    default_integer: int = None,
    default_sequence: Union[TList[int], Tuple[int, ...]] = None,
    allow_none=False,
    non_negative: bool = True,
    description="",
):
    """Returns a dataclass field with marshmallow metadata enforcing numeric inputs or a tuple of numeric
    inputs."""

    class IntegerOrIntegerSequenceField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, int):
                if non_negative:
                    if value < 0:
                        raise ValidationError("Value must be positive.")
                return value
            if isinstance(value, (tuple, list)):
                if non_negative:
                    for v in value:
                        if v < 0:
                            raise ValidationError("Values must be positive.")
                return value
            raise ValidationError("Field should be either an integer, tuple of integers, or a list of integers")

        def _jsonschema_type_mapping(self):
            numeric_option = {
                "type": "integer",
                "title": "integer_option",
                "default": default_integer,
                "description": "Set to a valid number.",
            }
            sequence_option = {
                "type": "array",
                "title": "sequence_option",
                "items": {"type": "number"},
                "default": default_sequence,
                "description": "Set to a valid number.",
            }

            oneof_list = [
                numeric_option,
                sequence_option,
            ]

            return {"oneOf": oneof_list, "title": self.name, "description": description, "default": default}

    return field(
        metadata={
            "marshmallow_field": IntegerOrIntegerSequenceField(
                allow_none=allow_none, load_default=default, dump_default=default, metadata={"description": description}
            )
        },
        default=default,
    )


def PositiveIntegerOrTupleOrStringOptions(
    options: TList[str] = None,
    allow_none=False,
    default: Union[None, int, Tuple[int, ...], str] = None,
    default_integer: Union[None, int] = None,
    default_tuple: Union[None, Tuple[int, ...]] = None,
    default_option: Union[None, str] = None,
    description="",
):
    """Returns a dataclass field with marshmallow metadata enforcing numeric inputs, a tuple of numeric inputs, or
    a string value."""

    class IntegerTupleStringOptionsField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, int):
                if value < 0:
                    raise ValidationError("Value must be positive.")
                return value
            if isinstance(value, tuple):
                for v in value:
                    if v < 0:
                        raise ValidationError("Values must be positive.")
                return value
            if isinstance(value, str):
                if value not in options:
                    raise ValidationError(f"String value should be one of {options}")
                return value

            raise ValidationError("Field should be either an integer, tuple of integers, or a string")

        def _jsonschema_type_mapping(self):
            if None in options and not self.allow_none:
                raise AssertionError(
                    f"Provided string options `{options}` includes `None`, but field is not set to allow `None`."
                )

            # Prepare numeric option:
            numeric_option = {
                "type": "integer",
                "title": "integer_option",
                "default": default_integer,
                "description": "Set to a valid number.",
            }
            tuple_option = {
                "type": "array",
                "title": "tuple_option",
                "items": [{"type": "number", "minimum": 0, "maximum": 999999}] * 2,
                "default": default_tuple,
                "description": "Set to a valid number.",
            }

            # Prepare string option (remove None):
            if None in options:
                options.remove(None)
            string_option = {
                "type": "string",
                "enum": options,
                "default": default_option,
                "title": "preconfigured_option",
                "description": "Choose a preconfigured option.",
            }
            oneof_list = [
                numeric_option,
                tuple_option,
                string_option,
            ]

            return {"oneOf": oneof_list, "title": self.name, "description": description, "default": default}

    return field(
        metadata={
            "marshmallow_field": IntegerTupleStringOptionsField(
                allow_none=allow_none, load_default=default, dump_default=default, metadata={"description": description}
            )
        },
        default=default,
    )


def Dict(default: Union[None, TDict] = None, description: str = "", parameter_metadata: ParameterMetadata = None):
    """Returns a dataclass field with marshmallow metadata enforcing input must be a dict."""
    if default is not None:
        try:
            assert isinstance(default, dict)
            assert all([isinstance(k, str) for k in default.keys()])
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    return field(
        metadata={
            "marshmallow_field": fields.Dict(
                fields.String(),
                allow_none=True,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default_factory=lambda: default,
    )


def List(
    list_type: Union[Type[str], Type[int], Type[float]] = str, default: Union[None, TList[Any]] = None, description=""
):
    """Returns a dataclass field with marshmallow metadata enforcing input must be a list."""
    if default is not None:
        try:
            assert isinstance(default, list)

        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")

    if list_type is str:
        field_type = fields.String()
    elif list_type is int:
        field_type = fields.Integer()
    elif list_type is float:
        field_type = fields.Float()
    else:
        raise ValueError(f"Invalid list type: `{list_type}`")

    return field(
        metadata={
            "marshmallow_field": fields.List(
                field_type,
                allow_none=True,
                load_default=default,
                dump_default=default,
                metadata={"description": description},
            )
        },
        default_factory=lambda: default,
    )


def DictList(
    default: Union[None, TList[TDict]] = None, description: str = "", parameter_metadata: ParameterMetadata = None
):
    """Returns a dataclass field with marshmallow metadata enforcing input must be a list of dicts."""
    if default is not None:
        try:
            assert isinstance(default, list)
            assert all([isinstance(d, dict) for d in default])
            for d in default:
                assert all([isinstance(k, str) for k in d.keys()])
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")

    return field(
        metadata={
            "marshmallow_field": fields.List(
                fields.Dict(fields.String()),
                allow_none=True,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default_factory=lambda: default,
    )


def Embed():
    """Returns a dataclass field with marshmallow metadata enforcing valid values for embedding input feature
    names.

    In particular, int and str values are allowed, and in the latter case the value must be one of the allowed
    `_embed_options`.
    """
    _embed_options = ["add"]

    class EmbedInputFeatureNameField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return value

            if isinstance(value, str):
                if value not in _embed_options:
                    raise ValidationError(f"Expected one of: {_embed_options}, found: {value}")
                return value

            if isinstance(value, int):
                return value

            raise ValidationError("Field should be int or str")

        def _jsonschema_type_mapping(self):
            return {
                "oneOf": [
                    {
                        "type": "string",
                        "enum": _embed_options,
                        "default": "add",
                        "title": "embed_string_option",
                        "description": "MISSING",
                    },
                    {"type": "integer", "title": "embed_integer_option", "description": "MISSING"},
                    {"type": "null", "title": "embed_null_option", "description": "MISSING"},
                ],
                "title": self.name,
                "description": "Valid options for embedding (or not embedding) input feature names.",
            }

    return field(
        metadata={
            "marshmallow_field": EmbedInputFeatureNameField(allow_none=True, load_default=None, dump_default=None)
        },
        default=None,
    )


def InitializerOrDict(default: str = "xavier_uniform", description: str = ""):
    """Returns a dataclass field with marshmallow metadata allowing customizable initializers.

    In particular, allows str or dict types; in the former case the field is equivalent to `InitializerOptions` while in
    the latter case a dict can be defined with the `type` field enforced to be one of `initializer_registry` as usual
    while additional properties are unrestricted.
    """
    initializers = list(initializer_registry.keys())
    if not isinstance(default, str) or default not in initializers:
        raise ValidationError(f"Invalid default: `{default}`")

    class InitializerOptionsOrCustomDictField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, str):
                if value not in initializers:
                    raise ValidationError(f"Expected one of: {initializers}, found: {value}")
                return value

            if isinstance(value, dict):
                if "type" not in value:
                    raise ValidationError("Dict must contain 'type'")
                if value["type"] not in initializers:
                    raise ValidationError(f"Dict expected key 'type' to be one of: {initializers}, found: {value}")
                return value

            raise ValidationError("Field should be str or dict")

        def _jsonschema_type_mapping(self):
            initializers = list(initializer_registry.keys())
            return {
                "oneOf": [
                    # Note: default not provided in the custom dict option:
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": initializers},
                        },
                        "required": ["type"],
                        "title": "initializer_custom_option",
                        "additionalProperties": True,
                        "description": "Customize an existing initializer.",
                    },
                    {
                        "type": "string",
                        "enum": initializers,
                        "default": default,
                        "title": "initializer_preconfigured_option",
                        "description": "Pick a preconfigured initializer.",
                    },
                ],
                "title": self.name,
                "default": default,
                "description": description,
            }

    return field(
        metadata={
            "marshmallow_field": InitializerOptionsOrCustomDictField(
                allow_none=False, load_default=default, dump_default=default, metadata={"description": description}
            )
        },
        default=default,
    )


def FloatRangeTupleDataclassField(
    n=2, default: Tuple = (0.9, 0.999), allow_none: bool = False, min=0, max=1, description=""
):
    """Returns a dataclass field with marshmallow metadata enforcing a `N`-dim.

    tuple with all values in given range. In particular, inputs must be N-dimensional tuples of purely numeric values
    within [min, max] range, i.e. inclusive. The generated JSON schema uses a restricted array type as the equivalent
    representation of a Python tuple.
    """
    if default is not None and n != len(default):
        raise ValidationError(f"Dimension of tuple '{n}' must match dimension of default val. '{default}'")

    class FloatTupleMarshmallowField(fields.Tuple):
        def _jsonschema_type_mapping(self):
            if default is not None:
                validate_range(default)
            return {
                "oneOf": [
                    {
                        "type": "array",
                        "items": [
                            {
                                "type": "number",
                                "minimum": min,
                                "maximum": max,
                            }
                        ]
                        * n,
                        "default": default,
                        "description": description,
                    },
                    {"type": "null", "title": "null_float_tuple_option", "description": "None"},
                ],
                "title": self.name,
                "default": default,
                "description": "Valid options for FloatRangeTupleDataclassField.",
            }

    def validate_range(data: Tuple):
        if isinstance(data, tuple) and all([isinstance(x, float) or isinstance(x, int) for x in data]):
            if all(list(map(lambda b: min <= b <= max, data))):
                return data
            raise ValidationError(
                f"Values in received tuple should be in range [{min},{max}], instead received: {data}"
            )
        raise ValidationError(f'Received value should be of {n}-dimensional "Tuple[float]", instead received: {data}')

    try:
        if default is not None:
            validate_range(default)
        if default is None and not allow_none:
            raise ValidationError("Default value must not be None if allow_none is False")
    except Exception:
        raise ValidationError(f"Invalid default: `{default}`")

    return field(
        metadata={
            "marshmallow_field": FloatTupleMarshmallowField(
                tuple_fields=[fields.Float()] * n,
                allow_none=allow_none,
                validate=validate_range,
                load_default=default,
                dump_default=default,
                metadata={"description": description},
            )
        },
        default=default,
    )


def OneOfOptionsField(
    default: Any, description: str, allow_none: bool, field_options: TList, parameter_metadata: ParameterMetadata = None
):
    """Returns a dataclass field that is a combination of the other fields defined in `ludwig.schema.utils`."""
    field_options_allow_none = any(option.metadata["marshmallow_field"].allow_none for option in field_options)

    class OneOfOptionsCombinatorialField(fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            if allow_none and value is None:
                return None
            for option in field_options:
                mfield_meta = option.metadata["marshmallow_field"]
                try:
                    if value is None and mfield_meta.allow_none:
                        return None
                    mfield_meta.validate(value)
                    return mfield_meta._serialize(value, attr, obj, **kwargs)
                except Exception:
                    continue
            raise ValidationError(f"Value to serialize does not match any valid option schemas: {value}")

        def _deserialize(self, value, attr, obj, **kwargs):
            if allow_none and value is None:
                return None
            for option in field_options:
                mfield_meta = option.metadata["marshmallow_field"]
                try:
                    mfield_meta.validate(value)
                    return mfield_meta._deserialize(value, attr, obj, **kwargs)
                except Exception:
                    continue
            raise ValidationError(f"Value to deserialize does not match any valid option schemas: {value}")

        def _jsonschema_type_mapping(self):
            """Constructs a oneOf schema by iteratively adding the schemas of `field_options` to a list."""
            oneOf = {"oneOf": [], "description": description, "default": default}

            for option in field_options:
                mfield_meta = option.metadata["marshmallow_field"]

                # If the option inherits from a custom dataclass-field, then use the custom jsonschema:
                if hasattr(mfield_meta, "_jsonschema_type_mapping"):
                    oneOf["oneOf"].append(mfield_meta._jsonschema_type_mapping())
                # Otherwise, extract the jsonschema using a dummy dataclass as intermediary:
                else:

                    @m_dataclass
                    class DummyClass:
                        tmp: Any = option

                    dummy_schema = unload_jsonschema_from_marshmallow_class(DummyClass)
                    tmp_json_schema = dummy_schema["properties"]["tmp"]
                    oneOf["oneOf"].append(tmp_json_schema)

            # Add null as an option if none of the field options allow none:
            oneOf["oneOf"] += (
                [{"type": "null", "title": "null_option", "description": "Disable this parameter."}]
                if allow_none and not field_options_allow_none
                else []
            )

            return oneOf

    # Create correct default kwarg to pass to dataclass field constructor:
    def is_primitive(value):
        primitive = (int, str, bool)
        return isinstance(value, primitive)

    default_kwarg = {}
    if is_primitive(default):
        default_kwarg["default"] = default
    else:
        default_kwarg["default_factory"] = lambda: default

    return field(
        metadata={
            "marshmallow_field": OneOfOptionsCombinatorialField(
                allow_none=allow_none, load_default=default, dump_default=default, metadata={"description": description}
            ),
            "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
        },
        **default_kwarg,
    )
