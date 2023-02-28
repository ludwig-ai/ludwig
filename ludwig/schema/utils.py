import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import field, Field
from typing import Any
from typing import Dict as TDict
from typing import List as TList
from typing import Optional, Set, Tuple, Type, TypeVar, Union

import marshmallow_dataclass
import yaml
from marshmallow import fields, INCLUDE, pre_load, schema, validate, ValidationError
from marshmallow.utils import missing
from marshmallow_dataclass import dataclass as m_dataclass
from marshmallow_jsonschema import JSONSchema as js

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ACTIVE, COLUMN, NAME, PROC_COLUMN, TYPE
from ludwig.modules.reduction_modules import reduce_mode_registry
from ludwig.schema.metadata import COMMON_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.utils.misc_utils import memoized_method
from ludwig.utils.registry import Registry
from ludwig.utils.torch_utils import activations, initializer_registry

RECURSION_STOP_ENUM = {"weights_initializer", "bias_initializer", "norm_params"}
ludwig_dataclass = m_dataclass(repr=False, order=True)


@DeveloperAPI
def get_marshmallow_field_class_name(field):
    """Returns a human-readable string of the marshmallow class name."""
    return field.metadata["marshmallow_field"].__class__.__name__


@DeveloperAPI
def load_config(cls: Type["BaseMarshmallowConfig"], **kwargs) -> "BaseMarshmallowConfig":  # noqa 0821
    """Takes a marshmallow class and instantiates it with the given keyword args as parameters."""
    assert_is_a_marshmallow_class(cls)
    schema = cls.Schema()
    return schema.load(kwargs)


@DeveloperAPI
def load_trainer_with_kwargs(
    model_type: str, kwargs: dict
) -> Tuple["BaseMarshmallowConfig", TDict[str, Any]]:  # noqa: F821
    """Special case of `load_config_with_kwargs` for the trainer schemas.

    In particular, it chooses the correct default type for an incoming config (if it doesn't have one already), but
    otherwise passes all other parameters through without change.
    """
    from ludwig.constants import MODEL_ECD
    from ludwig.schema.trainer import ECDTrainerConfig, GBMTrainerConfig

    trainer_schema = ECDTrainerConfig if model_type == MODEL_ECD else GBMTrainerConfig

    return load_config_with_kwargs(trainer_schema, kwargs)


@DeveloperAPI
def load_config_with_kwargs(
    cls: Type["BaseMarshmallowConfig"], kwargs_overrides
) -> Tuple["BaseMarshmallowConfig", TDict[str, Any]]:  # noqa 0821
    """Instatiates an instance of the marshmallow class and kwargs overrides instantiantes the schema.

    Returns a tuple of config, and a dictionary of any keys in kwargs_overrides which are not present in config.
    """
    assert_is_a_marshmallow_class(cls)
    schema = cls.Schema()
    fields = schema.fields.keys()
    return load_config(cls, **{k: v for k, v in kwargs_overrides.items() if k in fields}), {
        k: v for k, v in kwargs_overrides.items() if k not in fields
    }


@DeveloperAPI
def convert_submodules(config_dict: dict) -> TDict[str, any]:
    """Helper function for converting submodules to dictionaries during a config object to dict transformation.

    Args:
        config_dict: Top level config dictionary with un-converted submodules

    Returns:
        The fully converted config dictionary
    """
    output_dict = copy.deepcopy(config_dict)

    for k, v in output_dict.items():
        if isinstance(v, dict):
            convert_submodules(v)

        elif isinstance(v, BaseMarshmallowConfig):
            output_dict[k] = v.to_dict()
            convert_submodules(output_dict[k])

        elif isinstance(v, list):
            # Handle generic lists
            output_dict[k] = [x.to_dict() if isinstance(x, BaseMarshmallowConfig) else x for x in v]

        elif isinstance(v, ListSerializable):
            output_dict[k] = v.to_list()

        else:
            continue

    return output_dict


@DeveloperAPI
def create_cond(if_pred: TDict, then_pred: TDict):
    """Returns a JSONSchema conditional for the given if-then predicates."""
    return {
        "if": {"properties": {k: {"const": v} for k, v in if_pred.items()}},
        "then": {"properties": then_pred},
    }


@DeveloperAPI
def remove_duplicate_fields(properties: dict, fields: Optional[TList[str]] = None) -> None:
    """Util function for removing duplicated schema elements. For example, input feature json schema mapping has a
    type param defined directly on the json schema, but also has a parameter defined on the schema class. We need
    both -

    json schema level for validation and schema class level for config object - though we only need the json schema
    level for validation, so we get rid of the duplicates when converting to json schema.

    Args:
        properties: Dictionary of properties generated from a Ludwig schema class
    """
    duplicate_fields = [NAME, TYPE, COLUMN, PROC_COLUMN, ACTIVE] if fields is None else fields
    for key in duplicate_fields:  # TODO: Remove col/proc_col once train metadata decoupled
        if key in properties:
            del properties[key]


@DeveloperAPI
class ListSerializable(ABC):
    @abstractmethod
    def to_list(self) -> TList:
        pass


ConfigT = TypeVar("ConfigT", bound="BaseMarshmallowConfig")


@DeveloperAPI
class BaseMarshmallowConfig(ABC):
    """Base marshmallow class for common attributes and metadata."""

    class Meta:
        """Sub-class specifying meta information for Marshmallow.

        Currently only sets `unknown` flag to `EXCLUDE`. This is done to mirror Ludwig behavior: unknown properties are
        excluded from `load` calls so that the marshmallow_dataclass package can be used but
        `unload_jsonschema_from_marshmallow_class` will manually set a marshmallow schema's `additionalProperties` attr.
        to True so that JSON objects with extra properties do not raise errors; as a result properties are picked and
        filled in as necessary.
        """

        unknown = INCLUDE  # TODO: Change to RAISE and update descriptions once we want to enforce strict schemas.
        "Flag that sets marshmallow `load` calls to ignore unknown properties passed as a parameter."

        ordered = True
        "Flag that maintains the order of defined parameters in the schema"

    def to_dict(self):
        """Method for getting a dictionary representation of this dataclass.

        Returns: dict for this dataclass
        """
        return convert_submodules(self.__dict__)

    @pre_load
    def log_deprecation_warnings(self, data, **kwargs):
        leftover = copy.deepcopy(data)
        for key in data.keys():
            if key not in self.fields:
                del leftover[key]
                # `type` is not declared on most schemas and is instead added dynamically:
                if key != "type" and key != "feature_type":
                    warnings.warn(
                        f'"{key}" is not a valid parameter for the "{self.__class__.__name__}" schema, will be flagged '
                        "as an error in v0.8",
                        DeprecationWarning,
                    )
        return leftover

    @classmethod
    def from_dict(cls: Type[ConfigT], d: TDict[str, Any]) -> ConfigT:
        schema = cls.get_class_schema()()
        return schema.load(d)

    @classmethod
    @memoized_method(maxsize=1)
    def get_valid_field_names(cls) -> Set[str]:
        schema = cls.get_class_schema()()
        return set(schema.fields.keys())

    @classmethod
    @memoized_method(maxsize=1)
    def get_class_schema(cls):
        return marshmallow_dataclass.class_schema(cls)

    def __repr__(self):
        return yaml.dump(self.to_dict(), sort_keys=False)


@DeveloperAPI
def assert_is_a_marshmallow_class(cls):
    assert hasattr(cls, "Schema") and isinstance(
        cls.Schema, schema.SchemaMeta
    ), f"Expected marshmallow class, but `{cls}` does not have the necessary `Schema` attribute."


@DeveloperAPI
def unload_jsonschema_from_marshmallow_class(mclass, additional_properties: bool = True) -> TDict:
    """Helper method to directly get a marshmallow class's JSON schema without extra wrapping props."""
    assert_is_a_marshmallow_class(mclass)
    schema = js(props_ordered=True).dump(mclass.Schema())["definitions"][mclass.__name__]
    # Check top-level ParameterMetadata:
    for prop in schema["properties"]:
        prop_schema = schema["properties"][prop]
        if "parameter_metadata" in prop_schema:
            prop_schema["parameter_metadata"] = copy.deepcopy(prop_schema["parameter_metadata"])
    schema["additionalProperties"] = additional_properties
    return schema


@DeveloperAPI
def InitializerOptions(default: str = "xavier_uniform", description="", parameter_metadata: ParameterMetadata = None):
    """Utility wrapper that returns a `StringOptions` field with keys from `initializer_registry`."""
    return StringOptions(
        list(initializer_registry.keys()),
        default=default,
        allow_none=False,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def ActivationOptions(
    default: Union[str, None] = "relu", description=None, parameter_metadata: ParameterMetadata = None
):
    """Utility wrapper that returns a `StringOptions` field with keys from `activations` registry."""
    description = description or "Default activation function applied to the output of the fully connected layers."
    parameter_metadata = parameter_metadata or COMMON_METADATA["activation"]
    return StringOptions(
        list(activations.keys()),
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def ReductionOptions(default: Union[None, str] = None, description="", parameter_metadata: ParameterMetadata = None):
    """Utility wrapper that returns a `StringOptions` field with keys from `reduce_mode_registry`."""
    return StringOptions(
        list(reduce_mode_registry.keys()),
        default=default,
        allow_none=True,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def RegularizerOptions(
    default: Union[None, str],
    allow_none: bool = False,
    description="",
    parameter_metadata: ParameterMetadata = None,
):
    """Utility wrapper that returns a `StringOptions` field with prefilled regularizer options."""
    return StringOptions(
        ["l1", "l2", "l1_l2"],
        default=default,
        allow_none=allow_none,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def String(
    description: str,
    default: Union[None, str],
    allow_none: bool = False,
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
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            ),
            # "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
        },
        default=default,
    )


@DeveloperAPI
def StringOptions(
    options: TList[str],
    default: Union[None, str],
    allow_none: bool = False,
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


@DeveloperAPI
def ProtectedString(
    pstring: str,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Alias for a `StringOptions` field with only one option.

    Useful primarily for `type` parameters.
    """
    return StringOptions(
        options=[pstring],
        default=pstring,
        allow_none=False,
        description=description,
        parameter_metadata=parameter_metadata,
    )


@DeveloperAPI
def IntegerOptions(
    options: TList[int],
    default: Union[None, int],
    allow_none: bool = False,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata that enforces integer inputs must be one of `options`.

    By default, None is allowed (and automatically appended) to the allowed list of options.
    """
    # If None should be allowed for an enum field, it also has to be defined as a valid
    # [option](https://github.com/json-schema-org/json-schema-spec/issues/258):
    if len(options) <= 0:
        raise ValidationError("Must provide non-empty list of options!")
    if default is not None and not isinstance(default, int):
        raise ValidationError(f"Provided default `{default}` should be an int!")
    if allow_none and None not in options:
        options += [None]
    if not allow_none and None in options:
        options.remove(None)
    if default not in options:
        raise ValidationError(f"Provided default `{default}` is not one of allowed options: {options} ")
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
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


@DeveloperAPI
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
                # Necessary because marshmallow will otherwise cast any non-boolean value to a boolean:
                validate=validate.OneOf([True, False]),
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


@DeveloperAPI
def Integer(default: Union[None, int], allow_none=False, description="", parameter_metadata: ParameterMetadata = None):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs."""
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


@DeveloperAPI
def PositiveInteger(
    description: str, default: Union[None, int], allow_none: bool = False, parameter_metadata: ParameterMetadata = None
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs must be
    positive."""
    val = validate.Range(min=1)

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


@DeveloperAPI
def NonNegativeInteger(
    description: str,
    default: Union[None, int],
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs must be
    nonnegative."""
    val = validate.Range(min=0)

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


@DeveloperAPI
def IntegerRange(
    description: str,
    default: Union[None, int],
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
    min: int = None,
    max: int = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
):
    """Returns a dataclass field with marshmallow metadata strictly enforcing (non-float) inputs must be in range
    set by relevant keyword args."""
    val = validate.Range(min=min, max=max, min_inclusive=min_inclusive, max_inclusive=max_inclusive)

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


@DeveloperAPI
def NonNegativeFloat(
    default: Union[None, float],
    allow_none: bool = False,
    description: str = "",
    max: Optional[float] = None,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata enforcing numeric inputs must be nonnegative."""
    val = validate.Range(min=0.0, max=max)

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


@DeveloperAPI
def FloatRange(
    default: Union[None, float],
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


@DeveloperAPI
def Dict(
    default: Union[None, TDict] = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata enforcing input must be a dict."""
    allow_none = allow_none or default is None

    if default is not None:
        try:
            assert isinstance(default, dict)
            assert all([isinstance(k, str) for k in default.keys()])
        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    elif not allow_none:
        default = {}

    load_default = lambda: copy.deepcopy(default)
    return field(
        metadata={
            "marshmallow_field": fields.Dict(
                fields.String(),
                allow_none=allow_none,
                load_default=load_default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default_factory=load_default,
    )


@DeveloperAPI
def List(
    list_type: Union[Type[str], Type[int], Type[float], Type[list]] = str,
    default: Union[None, TList[Any]] = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field with marshmallow metadata enforcing input must be a list."""
    if default is not None:
        try:
            assert isinstance(default, list)

        except Exception:
            raise ValidationError(f"Invalid default: `{default}`")
    elif not allow_none:
        default = []

    if list_type is str:
        field_type = fields.String()
    elif list_type is int:
        field_type = fields.Integer()
    elif list_type is float:
        field_type = fields.Float()
    elif list_type is list:
        field_type = fields.List(fields.Float())
    else:
        raise ValueError(f"Invalid list type: `{list_type}`")

    load_default = lambda: copy.deepcopy(default)
    return field(
        metadata={
            "marshmallow_field": fields.List(
                field_type,
                allow_none=allow_none,
                load_default=load_default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default_factory=load_default,
    )


@DeveloperAPI
def DictList(
    default: Union[None, TList[TDict]] = None,
    allow_none: bool = True,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
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
    elif not allow_none:
        default = []

    load_default = lambda: copy.deepcopy(default)
    return field(
        metadata={
            "marshmallow_field": fields.List(
                fields.Dict(fields.String()),
                allow_none=True,
                load_default=load_default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default_factory=load_default,
    )


@DeveloperAPI
def Embed(description: str = "", parameter_metadata: ParameterMetadata = None):
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
            "marshmallow_field": EmbedInputFeatureNameField(
                allow_none=True,
                load_default=None,
                dump_default=None,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=None,
    )


@DeveloperAPI
def InitializerOrDict(
    default: str = "xavier_uniform", description: str = "", parameter_metadata: ParameterMetadata = None
):
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
            param_metadata = convert_metadata_to_json(parameter_metadata) if parameter_metadata else None
            return {
                "oneOf": [
                    # Note: default not provided in the custom dict option:
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": initializers},
                        },
                        "required": ["type"],
                        "title": f"{self.name}_custom_option",
                        "additionalProperties": True,  # Will be removed by initializer refactor PR.
                        "description": "Customize an existing initializer.",
                        "parameter_metadata": param_metadata,
                    },
                    {
                        "type": "string",
                        "enum": initializers,
                        "default": default,
                        "title": f"{self.name}_preconfigured_option",
                        "description": "Pick a preconfigured initializer.",
                        "parameter_metadata": param_metadata,
                    },
                ],
                "title": self.name,
                "default": default,
                "description": description,
            }

    return field(
        metadata={
            "marshmallow_field": InitializerOptionsOrCustomDictField(
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


@DeveloperAPI
def FloatRangeTupleDataclassField(
    n: int = 2,
    default: Union[Tuple, None] = (0.9, 0.999),
    allow_none: bool = False,
    min: Union[int, None] = 0,
    max: Union[int, None] = 1,
    description: str = "",
    parameter_metadata: ParameterMetadata = None,
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
            items_schema = {"type": "number"}
            if min is not None:
                items_schema["minimum"] = min
            if max is not None:
                items_schema["maximum"] = max
            one_of = [
                {
                    "type": "array",
                    "items": [{**items_schema}] * n,
                    "default": default,
                    "description": description,
                },
            ]
            if allow_none:
                one_of.append({"type": "null", "title": "null_float_tuple_option", "description": "None"})
            return {
                "oneOf": one_of,
                "title": self.name,
                "default": default,
                "description": "Valid options for FloatRangeTupleDataclassField.",
            }

    def validate_range(data: Tuple):
        if isinstance(data, tuple) and all([isinstance(x, float) or isinstance(x, int) for x in data]):
            minmax_checks = []
            if min is not None:
                minmax_checks += list(map(lambda b: min <= b, data))
            if max is not None:
                minmax_checks += list(map(lambda b: b <= max, data))
            if all(minmax_checks):
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
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            )
        },
        default=default,
    )


@DeveloperAPI
def OneOfOptionsField(
    default: Any,
    description: str,
    field_options: TList,
    allow_none: bool = False,
    parameter_metadata: ParameterMetadata = None,
):
    """Returns a dataclass field that is a combination of the other fields defined in `ludwig.schema.utils`.

    NOTE: There can be at most one field_option with `allow_none=True`, or else a None value can be attributed to
    multiple field_options, which this JSON validator does not permit.
    """
    if default is None:
        # If the default is None, then this field allows none.
        allow_none = True

    fields_that_allow_none = [option for option in field_options if option.metadata["marshmallow_field"].allow_none]
    if len(fields_that_allow_none) > 1 and allow_none:
        raise ValueError(
            f"The governing OneOf has allow_none=True, but there are some field options that themselves "
            "allow_none=True, which is ambiguous for JSON validation. To maintain allow_none=True for the overall "
            "field, add allow_none=False to each of the field_options: "
            f"{[get_marshmallow_field_class_name(field) for field in fields_that_allow_none]}, and rely on the "
            "governing OneOf's allow_none=True to set the allow_none policy."
        )

    if fields_that_allow_none and not allow_none:
        raise ValueError(
            "The governing OneOf has allow_none=False, while None is permitted by the following field_options: "
            f"{[get_marshmallow_field_class_name(field) for field in fields_that_allow_none]}. This is contradictory. "
            "Please set allow_none=False for each field option to make this consistent."
        )

    class OneOfOptionsCombinatorialField(fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            if allow_none and value is None:
                return None
            for option in field_options:
                mfield_meta = option.metadata["marshmallow_field"]
                try:
                    if value is None and mfield_meta.allow_none:
                        return None
                    # Not every field (e.g. our custom dataclass fields) has a `validate` method:
                    if mfield_meta.validate:
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
                    # Not every field (e.g. our custom dataclass fields) has a `validate` method:
                    if mfield_meta.validate:
                        mfield_meta.validate(value)
                    return mfield_meta._deserialize(value, attr, obj, **kwargs)
                except Exception:
                    continue
            raise ValidationError(f"Value to deserialize does not match any valid option schemas: {value}")

        def _jsonschema_type_mapping(self):
            """Constructs a oneOf schema by iteratively adding the schemas of `field_options` to a list."""
            oneOf = {
                "oneOf": [],
                "description": description,
                "default": default,
                "title": self.name,
                "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
            }

            for idx, option in enumerate(field_options):
                mfield_meta = option.metadata["marshmallow_field"]

                # Necessary for key/name de-duplication in case a name is not supplied by the user:
                mfield_meta_class_name = str(mfield_meta.__class__).split(".")[-1].split("'")[0].lower()

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
                    # Manually set the title, otherwise it would be 'tmp':
                    tmp_json_schema["title"] = f"{self.name}_{mfield_meta_class_name}_option"
                    oneOf["oneOf"].append(tmp_json_schema)

            # Add null as an option if we want to allow none but none of the field options allow none.
            any_field_options_allow_none = any(
                option.metadata["marshmallow_field"].allow_none for option in field_options
            )
            if allow_none and not any_field_options_allow_none:
                oneOf["oneOf"] += [{"type": "null", "title": "null_option", "description": "Disable this parameter."}]

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
                allow_none=allow_none,
                load_default=default,
                dump_default=default,
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(parameter_metadata) if parameter_metadata else None,
                },
            ),
        },
        **default_kwarg,
    )


class TypeSelection(fields.Field):
    def __init__(
        self,
        registry: Registry,
        default_value: Optional[str] = None,
        key: str = "type",
        description: str = "",
        parameter_metadata: ParameterMetadata = None,
    ):
        self.registry = registry
        self.default_value = default_value
        self.key = key

        dump_default = missing
        load_default = missing
        self.default_factory = None
        if self.default_value is not None:
            default_obj = {key: default_value}
            cls = self.get_schema_from_registry(self.default_value.lower())
            self.default_factory = lambda: cls.Schema().load(default_obj)
            load_default = self.default_factory
            dump_default = cls.Schema().dump(default_obj)

        super().__init__(
            allow_none=False,
            dump_default=dump_default,
            load_default=load_default,
            metadata={"description": description, "parameter_metadata": convert_metadata_to_json(parameter_metadata)},
        )

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return None
        if isinstance(value, dict):
            cls_type = value.get(self.key)
            cls_type = cls_type.lower() if cls_type else self.default_value
            if cls_type in self.registry:
                cls = self.get_schema_from_registry(cls_type)
                try:
                    return cls.Schema().load(value)
                except (TypeError, ValidationError) as e:
                    raise ValidationError(f"Invalid params: {value}, see `{cls}` definition") from e
            raise ValidationError(f"Invalid type: '{cls_type}', expected one of: {list(self.registry.keys())}")
        raise ValidationError(f"Invalid param {value}, expected `None` or `dict`")

    def get_schema_from_registry(self, key: str) -> Type[BaseMarshmallowConfig]:
        return self.registry[key]

    def get_default_field(self) -> Field:
        default_factory = lambda: None
        if self.default_factory is not None:
            default_factory = self.default_factory

        return field(
            metadata={"marshmallow_field": self},
            default_factory=default_factory,
        )


@DeveloperAPI
class DictMarshmallowField(fields.Field):
    def __init__(
        self,
        cls: Type[BaseMarshmallowConfig],
        allow_none: bool = True,
        default_missing: bool = False,
        description: str = "",
    ):
        self.cls = cls

        dump_default = missing
        load_default = missing
        self.default_factory = None
        if not default_missing:
            default_obj = {}
            self.default_factory = lambda: cls.Schema().load(default_obj)
            load_default = self.default_factory
            dump_default = cls.Schema().dump(default_obj)

        super().__init__(
            allow_none=allow_none,
            dump_default=dump_default,
            load_default=load_default,
            metadata={"description": description},
        )

    def _deserialize(self, value, attr, data, **kwargs):
        if value is None:
            return value
        if isinstance(value, dict):
            try:
                return self.cls.Schema().load(value)
            except (TypeError, ValidationError) as e:
                # TODO(travis): this seems much too verbose, does the validation error not show the specific error?
                raise ValidationError(f"Invalid params: {value}, see `{self.cls}` definition. Error: {e}")
        raise ValidationError(f"Invalid param {value}, expected `None` or `dict`")

    def get_default_field(self) -> Field:
        default_factory = lambda: None
        if self.default_factory is not None:
            default_factory = self.default_factory

        return field(
            metadata={"marshmallow_field": self},
            default_factory=default_factory,
        )
