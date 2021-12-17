from dataclasses import field

import marshmallow_dataclass
from marshmallow import fields, validate, ValidationError

from ludwig.constants import TYPE
from ludwig.modules.optimization_modules import optimizers_registry
from ludwig.modules.reduction_modules import reduce_mode_registry
from ludwig.utils.torch_utils import initializer_registry


def InitializerOptions(default=None):
    return StringOptions(list(initializer_registry.keys()), default=default, nullable=True)


def ReductionOptions(default=None):
    return StringOptions(
        list(reduce_mode_registry.keys()),
        default=default,
        nullable=True,
    )


def RegularizerOptions(nullable=True):
    return StringOptions(["l1", "l2", "l1_l2"], nullable=nullable)


def StringOptions(options, default=None, nullable=True):
    return field(
        metadata={
            "marshmallow_field": fields.String(
                validate=validate.OneOf(options),
                allow_none=nullable,
            )
        },
        default=default,
    )


def PositiveInteger(default=None):
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                validate=validate.Range(min=1),
                allow_none=default is None,
            )
        },
        default=default,
    )


def NonNegativeInteger(default=None):
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                validate=validate.Range(min=0),
                allow_none=True,
            )
        },
        default=default,
    )


def FloatRange(default=None, **kwargs):
    return field(
        metadata={
            "marshmallow_field": fields.Float(
                validate=validate.Range(**kwargs),
                allow_none=default is None,
            )
        },
        default=default,
    )


def DictList():
    return field(
        metadata={
            "marshmallow_field": fields.List(
                fields.Dict(fields.String()),
                allow_none=True,
            )
        },
        default=None,
    )


def Dict():
    return field(
        metadata={
            "marshmallow_field": fields.Dict(
                fields.String(),
                allow_none=True,
            )
        },
        default=None,
    )


def Embed():
    return field(metadata={"marshmallow_field": EmbedInputFeatureNameField(allow_none=True)}, default=None)


_embed_options = ["add"]


def InitializerOrDict(default="xavier_uniform"):
    return field(metadata={"marshmallow_field": InitializerOptionsOrCustomDictField(allow_none=False)}, default=default)


def OptimizerOptions(default={TYPE: "adam"}):
    return field(metadata={"marshmallow_field": OptimizerOptionsField(allow_none=False)}, default=default)


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
        return {"oneOf": [{"type": "string", "enum": _embed_options}, {"type": "integer"}, {"type": "null"}]}


class InitializerOptionsOrCustomDictField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        initializers = list(initializer_registry.keys())
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
                {"type": "string", "enum": initializers},
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": initializers},
                    },
                    "required": ["type"],
                    "additionalProperties": True,
                },
            ]
        }


class OptimizerOptionsField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        optimizers = list(optimizers_registry.keys())
        if isinstance(value, dict):
            if TYPE not in value:
                raise ValidationError(f"Dict must contain '{TYPE}' as key")
            if value[TYPE] not in optimizers:
                raise ValidationError(f"Dict expected key 'type' to be one of: {optimizers}, found: {value}")
            return value

        raise ValidationError(f"Field should be dict with '{TYPE}' property")

    def _jsonschema_type_mapping(self):
        optimizers = list(optimizers_registry.keys())
        return {
            "type": "object",
            "properties": {
                TYPE: {"type": "string", "enum": optimizers},
            },
            "required": ["type"],
            "additionalProperties": True,
        }


def load_config(cls, **kwargs):
    schema = marshmallow_dataclass.class_schema(cls)()
    return schema.load(kwargs)


def load_config_with_kwargs(cls, kwargs):
    schema = marshmallow_dataclass.class_schema(cls)()
    fields = schema.fields.keys()
    return load_config(cls, **{k: v for k, v in kwargs.items() if k in fields}), {
        k: v for k, v in kwargs.items() if k not in fields
    }
