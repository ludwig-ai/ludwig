from dataclasses import field
from typing import List, Tuple, Union

import marshmallow_dataclass
from marshmallow import fields, validate, ValidationError

from ludwig.modules.reduction_modules import reduce_mode_registry
from ludwig.utils.torch_utils import initializer_registry


def InitializerOptions(default: Union[None, str] = None):
    return StringOptions(list(initializer_registry.keys()), default=default, nullable=True)


def ReductionOptions(default: Union[None, str] = None):
    return StringOptions(
        list(reduce_mode_registry.keys()),
        default=default,
        nullable=True,
    )


def RegularizerOptions(nullable: bool = True):
    return StringOptions(["l1", "l2", "l1_l2"], nullable=nullable)


def StringOptions(options: List[str], default: Union[None, str] = None, nullable: bool = True):
    return field(
        metadata={
            "marshmallow_field": fields.String(
                validate=validate.OneOf(options),
                allow_none=nullable,
            )
        },
        default=default,
    )


def PositiveInteger(default: Union[None, int] = None):
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                validate=validate.Range(min=1),
                allow_none=default is None,
            )
        },
        default=default,
    )


def NonNegativeInteger(default: Union[None, int] = None):
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                validate=validate.Range(min=0),
                allow_none=True,
            )
        },
        default=default,
    )


def IntegerRange(default: Union[None, int] = None, **kwargs):
    return field(
        metadata={
            "marshmallow_field": fields.Integer(
                validate=validate.Range(**kwargs),
                allow_none=default is None,
            )
        },
        default=default,
    )


def NonNegativeFloat(default: Union[None, float] = None, **kwargs):
    return field(
        metadata={
            "marshmallow_field": fields.Float(
                validate=validate.Range(min=0.0),
                allow_none=default is None,
            )
        },
        default=default,
    )


def FloatRange(default: Union[None, float] = None, **kwargs):
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


def InitializerOrDict(default: str = "xavier_uniform"):
    return field(metadata={"marshmallow_field": InitializerOptionsOrCustomDictField(allow_none=False)}, default=default)


def FloatRangeTupleDataclassField(N=2, default: Tuple = (0.9, 0.999), min=0, max=1):
    if N != len(default):
        raise ValidationError(f"Dimension of tuple '{N}' must match dimension of default val. '{default}'")

    class FloatTupleMarshmallowField(fields.Tuple):
        def _jsonschema_type_mapping(self):
            return {
                "type": "array",
                "prefixItems": [
                    {
                        "type": "number",
                        "minimum": min,
                        "maximum": max,
                    }
                ]
                * N,
            }

    def validateRange(data: Tuple):
        if isinstance(data, tuple) and list(map(type, data)) == [float] * N:
            if all(list(map(lambda b: min <= b <= max, data))):
                return data
            raise ValidationError(
                f"Values in received tuple should be in range [{min},{max}], instead received: {data}"
            )
        raise ValidationError(f'Received value should be of {N}-dimensional "Tuple[float]", instead received: {data}')

    return field(
        metadata={
            "marshmallow_field": FloatTupleMarshmallowField(
                tuple_fields=[fields.Float()] * N,
                allow_none=default is None,
                validate=validateRange,
            )
        },
        default=default,
    )


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


def IntegerOrStringOptionsField(
    options: List[str],
    nullable: bool,
    default: Union[None, int],
    isIntegeric: bool = True,
    min: Union[None, int] = None,
    max: Union[None, int] = None,
    exclusiveMin: Union[None, int] = None,
    exclusiveMax: Union[None, int] = None,
):
    isIntegeric = True
    return NumericOrStringOptionsField(**locals())


def NumericOrStringOptionsField(
    options: List[str],
    nullable: bool,
    default: Union[None, int, float],
    isIntegeric: bool = False,
    min: Union[None, int] = None,
    max: Union[None, int] = None,
    exclusiveMin: Union[None, int] = None,
    exclusiveMax: Union[None, int] = None,
):
    class IntegerOrStringOptionsField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            msgType = "integer" if isIntegeric else "numeric"
            if (isIntegeric and isinstance(value, int)) or isinstance(value, float):
                if (
                    (min is not None and value < min)
                    or (exclusiveMin is not None and value <= exclusiveMin)
                    or (max is not None and value > max)
                    or (exclusiveMax is not None and value >= exclusiveMax)
                ):
                    errMinR, errMinN = "(", exclusiveMin if exclusiveMin is not None else "[", min
                    errMaxR, errMaxN = ")", exclusiveMax if exclusiveMax is not None else "]", max
                    raise ValidationError(
                        f"If value is {msgType} should be in range: {errMinR}{errMinN},{errMaxN}{errMaxR}"
                    )
                return value
            if isinstance(value, str):
                if value not in options:
                    raise ValidationError(f"String value should be one of {options}")
                return value

            raise ValidationError(f"Field should be either a {msgType} or string")

        def _jsonschema_type_mapping(self):
            jsonType = "integer" if isIntegeric else "number"
            tmp = {"type": jsonType}
            if min is not None:
                tmp["minimum"] = min
            if exclusiveMin is not None:
                tmp["exclusiveMinimum"] = exclusiveMin
            if max is not None:
                tmp["maximum"] = max
            if exclusiveMax is not None:
                tmp["exclusiveMaximum"] = exclusiveMax
            return {
                "oneOf": [
                    tmp,
                    {"type": "string", "enum": options},
                ]
            }

    return field(metadata={"marshmallow_field": IntegerOrStringOptionsField(allow_none=nullable)}, default=default)


def load_config(cls, **kwargs):
    schema = marshmallow_dataclass.class_schema(cls)()
    return schema.load(kwargs)


def load_config_with_kwargs(cls, kwargs):
    schema = marshmallow_dataclass.class_schema(cls)()
    fields = schema.fields.keys()
    return load_config(cls, **{k: v for k, v in kwargs.items() if k in fields}), {
        k: v for k, v in kwargs.items() if k not in fields
    }
