import re
from dataclasses import field
from typing import Dict, List, Tuple, Type, Union

import marshmallow_dataclass
from marshmallow import fields, Schema, validate, ValidationError
from marshmallow_jsonschema import JSONSchema as js
from pytkdocs import loader as pytkloader

from ludwig.modules.reduction_modules import reduce_mode_registry
from ludwig.utils.torch_utils import initializer_registry

restloader = pytkloader.Loader(docstring_style="restructured-text")
googleloader = pytkloader.Loader(docstring_style="google")


def create_cond(if_pred: Dict, then_pred: Dict):
    """Returns a JSONSchema conditional for the given if-then predicates."""
    return {
        "if": {"properties": {k: {"const": v} for k, v in if_pred.items()}},
        "then": {"properties": {k: v for k, v in then_pred.items()}},
    }


def get_fully_qualified_class_name(cls):
    """Returns fully dot-qualified path of a class, e.g. `ludwig.models.trainer.TrainerConfig` given
    `TrainerConfig`."""
    return ".".join([cls.__module__, cls.__name__])


def unload_schema_from_marshmallow_jsonschema_dump(mclass) -> Dict:
    """Helper method to get a marshmallow class's direct schema without extra wrapping props."""
    return js().dump(mclass.Schema())["definitions"][mclass.__name__]


def get_custom_schema_from_marshmallow_class(mclass: Type[Schema]) -> Dict:
    """Get Ludwig-customized schema from a given marshmallow class."""

    def cleanup_python_comment(dstring: str) -> str:
        """Cleans up some common issues with parsed comments/docstrings."""
        if dstring is None or dstring == "" or str.isspace(dstring):
            return ""
        # Add spaces after periods:
        dstring = re.sub(r"\.(?! )", ". ", dstring)
        # Replace internal newlines with spaces:
        dstring = re.sub("\n+", " ", dstring)
        # Replace any multiple-spaces with single spaces:
        dstring = re.sub(" +", " ", dstring)
        # Remove leading/ending spaces:
        dstring = dstring.strip()
        # Add final period if it's not there:
        dstring += "." if dstring[-1] != "." else ""
        # Capitalize first word in string and first word in each sentence.
        dstring = re.sub(r"((?<=[\.\?!]\s)(\w+)|(^\w+))", lambda m: m.group().capitalize(), dstring)
        return dstring

    def generate_extra_json_schema_props(schema_cls) -> Dict:
        """Workaround for adding 'description' and 'default' fields to a marshmallow schema's JSON Schema. Heres an
        extra description.

        Currently targeted for use with optimizer and combiner schema; if there is no description provided for a
        particular field, the description is pulled from the corresponding torch optimizer. Note that this currently
        overrides whatever may already be in the description/default fields. TODO(ksbrar): Watch this
        [issue](https://github.com/fuhrysteve/marshmallow-jsonschema/issues/41) to improve this eventually.
        """

        schema_dump = unload_schema_from_marshmallow_jsonschema_dump(schema_cls)
        schema_default = schema_cls()
        if schema_cls.__doc__ is not None:
            parsed_documentation = restloader.get_object_documentation(get_fully_qualified_class_name(schema_cls))

            # Add the top-level description to the schema if it exists:
            if parsed_documentation.docstring is not None:
                schema_dump["description"] = cleanup_python_comment(parsed_documentation.docstring)

            parsed_attrs = {attr.name: attr for attr in parsed_documentation.attributes}

            # For each prop in the schema, set its description and default if they are not already set. If not already
            # set and there is no available value from the Ludwig docstring, attempt to pull from PyTorch, if applicable
            # (e.g. for optimizer parameters).
            parsed_torch = (
                {
                    param.name: param
                    for param in googleloader.get_object_documentation(
                        get_fully_qualified_class_name(schema_cls.torch_type)
                    )
                    .docstring_sections[1]
                    .value
                }
                if hasattr(schema_cls, "torch_type") and schema_cls.torch_type is not None
                else None
            )
            for prop in schema_dump["properties"]:
                schema_prop = schema_dump["properties"][prop]

                if prop in parsed_attrs:
                    # Handle descriptions:

                    # Get the particular attribute's docstring (if it has one), strip the default from the string:
                    parsed_desc = parsed_attrs[prop].docstring
                    if parsed_desc is None:
                        parsed_desc = ""
                    parsed_desc = parsed_desc.split("(default: ")[0]

                    # If no description is provided, attempt to pull from torch if applicable:
                    desc = parsed_desc
                    if (
                        desc == ""
                        and parsed_torch is not None
                        and prop in parsed_torch
                        and (parsed_torch[prop].description is not None or parsed_torch[prop].description != "")
                    ):
                        desc = parsed_torch[prop].description

                    schema_prop["description"] = cleanup_python_comment(desc)

                    # Handle defaults:
                    if hasattr(schema_default, prop):
                        default = getattr(schema_default, prop)

                        # If the prop is itself another schema class, then convert its value to a dict:
                        if hasattr(type(default), "Schema"):
                            default = type(default).Schema().dump(default)

                        schema_prop["default"] = default
        return schema_dump

    return generate_extra_json_schema_props(mclass)


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

    def validate_range(data: Tuple):
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
                validate=validate_range,
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
    is_integer: bool = True,
    min: Union[None, int] = None,
    max: Union[None, int] = None,
    min_exclusive: Union[None, int] = None,
    max_exclusive: Union[None, int] = None,
):
    is_integer = True
    return NumericOrStringOptionsField(**locals())


def NumericOrStringOptionsField(
    options: List[str],
    nullable: bool,
    default: Union[None, int, float],
    is_integer: bool = False,
    min: Union[None, int] = None,
    max: Union[None, int] = None,
    min_exclusive: Union[None, int] = None,
    max_exclusive: Union[None, int] = None,
):
    class IntegerOrStringOptionsField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            msg_type = "integer" if is_integer else "numeric"
            if (is_integer and isinstance(value, int)) or isinstance(value, float):
                if (
                    (min is not None and value < min)
                    or (min_exclusive is not None and value <= min_exclusive)
                    or (max is not None and value > max)
                    or (max_exclusive is not None and value >= max_exclusive)
                ):
                    err_min_r, err_min_n = "(", min_exclusive if min_exclusive is not None else "[", min
                    errMaxR, errMaxN = ")", max_exclusive if max_exclusive is not None else "]", max
                    raise ValidationError(
                        f"If value is {msg_type} should be in range: {err_min_r}{err_min_n},{errMaxN}{errMaxR}"
                    )
                return value
            if isinstance(value, str):
                if value not in options:
                    raise ValidationError(f"String value should be one of {options}")
                return value

            raise ValidationError(f"Field should be either a {msg_type} or string")

        def _jsonschema_type_mapping(self):
            jsonType = "integer" if is_integer else "number"
            tmp = {"type": jsonType}
            if min is not None:
                tmp["minimum"] = min
            if min_exclusive is not None:
                tmp["exclusiveMinimum"] = min_exclusive
            if max is not None:
                tmp["maximum"] = max
            if max_exclusive is not None:
                tmp["exclusiveMaximum"] = max_exclusive
            oneOf = [
                tmp,
                {"type": "string", "enum": options},
            ]
            if nullable:
                oneOf += [{"type": "null"}]
            return {"oneOf": oneOf}

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
