from abc import ABC
from dataclasses import field
from typing import Any, ClassVar, Dict

from marshmallow import fields, ValidationError
from torch import nn

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

initializer_registry = Registry()


@DeveloperAPI
def register_initializer(name: str):
    def wrap(initializer_config: InitializerConfig):
        initializer_registry[name] = (initializer_config.initializer_fn, initializer_config)
        return initializer_config

    return wrap


@DeveloperAPI
def get_initialize_cls(name: str):
    """Get the initializer schema class from the initializer schema class registry."""
    return initializer_registry[name][1]


@DeveloperAPI
@ludwig_dataclass
class InitializerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for initializers. Not meant to be used directly."""

    initializer_fn: ClassVar
    "Class variable pointing to the corresponding initializer function."

    type: str
    """Name corresponding to an initializer.

    Technically mutable, but attempting to load a derived initializer with `type` set to a mismatched value will 
    result in a `ValidationError`.
    """

    def initializer_params(self) -> Dict[str, Any]:
        """Returns all params for this initializers without meta params."""
        params = self.to_dict()
        params.pop("initializer_fn", None)
        params.pop("type", None)
        return params


@DeveloperAPI
@register_initializer(name="zeros")
@ludwig_dataclass
class ZerosInitializer(InitializerConfig):
    """Zeros initialization."""

    initializer_fn: ClassVar = nn.init.zeros_

    type: str = schema_utils.ProtectedString("zeros")


@DeveloperAPI
@register_initializer(name="uniform")
@ludwig_dataclass
class UniformInitializer(InitializerConfig):
    """Uniform initialization."""

    initializer_fn: ClassVar = nn.init.uniform_

    type: str = schema_utils.ProtectedString("uniform")

    a: float = schema_utils.NonNegativeFloat(default=0.0, description="The lower bound of the uniform distribution")

    b: float = schema_utils.NonNegativeFloat(default=0.0, description="The upper bound of the uniform distribution")


@DeveloperAPI
@register_initializer(name="normal")
@ludwig_dataclass
class NormalInitializer(InitializerConfig):
    """Uniform initialization."""

    initializer_fn: ClassVar = nn.init.normal_

    type: str = schema_utils.ProtectedString("normal")

    mean: float = schema_utils.NonNegativeFloat(default=0.0, description="The mean of the normal distribution")

    std: float = schema_utils.NonNegativeFloat(
        default=1.0, description="The standard deviation of the normal distribution"
    )


@DeveloperAPI
def InitializerDataclassField(default, description="", parameter_metadata=None):
    """Custom dataclass field that when used inside of a dataclass will allow any initializer.

    :param default: Str or Dict specifying an initializer with a `type` field and its associated parameters. Will
            attempt to load an initializer with given params.
    :return: Initialized dataclass field that converts untyped dicts with params to initializer dataclass instances.
    """

    class InitializerMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict to a valid initializer and creates a corresponding
        `oneOf` JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None

            if isinstance(value, str):
                # If user provided the value as a string, assume they were providing the type
                value = {"type": value}

            if isinstance(value, dict):
                init_type = value.get("type")
                init_type = init_type.lower() if init_type else init_type
                if init_type in initializer_registry:
                    initializer = initializer_registry[init_type][1]
                    try:
                        return initializer.Schema().load(value)
                    except (TypeError, ValidationError) as e:
                        raise ValidationError(
                            f"Invalid params for initializer: {value}, see `{initializer}` definition"
                        ) from e
                raise ValidationError(
                    f"Invalid initializer type: '{init_type}', expected one of: {list(initializer_registry.keys())}."
                )
            raise ValidationError(f"Invalid initializer param {value}, expected `None` or `dict`")

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": list(initializer_registry.keys()),
                                    "default": default["type"],
                                    "description": "The type of initializer to use during the learning process",
                                },
                            },
                            "allOf": get_initializer_conds(),
                            "required": ["type"],
                        },
                        {
                            "type": "string",
                            "enum": list(initializer_registry.keys()),
                            "default": default["type"],
                            "description": "The type of initializer to use during the learning process",
                        },
                    ],
                },
                "title": "initializer_options",
                "description": description,
            }

    if isinstance(default, str):
        # If user provided the default as a string, assume they were providing the type
        default = {"type": default}

    if not isinstance(default, dict) or "type" not in default or default["type"] not in initializer_registry:
        raise ValidationError(f"Invalid default: `{default}`")

    try:
        initializer = initializer_registry[default["type"].lower()][1]
        load_default = lambda: initializer.Schema().load(default)
        dump_default = initializer.Schema().dump(default)

        # TODO(travis): use parameter_metadata
        return field(
            metadata={
                "marshmallow_field": InitializerMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                    metadata={"description": description},
                )
            },
            default_factory=load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported initializer type: {default['type']}. See initializer_registry. Details: {e}"
        )


@DeveloperAPI
def get_initializer_conds():
    """Returns a JSON schema of conditionals to validate against initializer types."""
    conds = []
    for initializer in initializer_registry:
        initializer_cls = initializer_registry[initializer][1]
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(initializer_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": initializer},
            other_props,
        )
        conds.append(preproc_cond)
    return conds
