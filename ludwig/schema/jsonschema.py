"""Minimal marshmallow-to-JSON-Schema converter.

Replaces the unmaintained ``marshmallow_jsonschema`` library which depends on
the removed ``pkg_resources`` package (setuptools 82+).
"""

from collections import OrderedDict

from marshmallow import fields, Schema, validate
from marshmallow.utils import missing

# Marshmallow field class -> JSON Schema type mapping
_FIELD_TYPE_MAP = {
    fields.String: {"type": "string"},
    fields.Integer: {"type": "integer"},
    fields.Float: {"type": "number", "format": "float"},
    fields.Number: {"type": "number"},
    fields.Boolean: {"type": "boolean"},
    fields.UUID: {"type": "string", "format": "uuid"},
    fields.DateTime: {"type": "string", "format": "date-time"},
    fields.Date: {"type": "string", "format": "date"},
    fields.Time: {"type": "string", "format": "time"},
    fields.TimeDelta: {"type": "string"},
    fields.Email: {"type": "string", "format": "email"},
    fields.Url: {"type": "string", "format": "uri"},
    fields.IP: {"type": "string", "format": "ipv4"},
    fields.IPInterface: {"type": "string"},
    fields.Decimal: {"type": "number", "format": "decimal"},
    fields.Raw: {"type": "string"},
    fields.Dict: {"type": "object"},
    fields.List: {"type": "array"},
    fields.Tuple: {"type": "array"},
}


def _resolve_json_type(field_cls):
    """Walk the MRO to find a JSON type for a marshmallow field class."""
    for cls in field_cls.__mro__:
        if cls in _FIELD_TYPE_MAP:
            return dict(_FIELD_TYPE_MAP[cls])
    return {"type": "string"}


def _field_to_jsonschema(field_obj, parent_schema=None):
    """Convert a single marshmallow field to a JSON Schema fragment."""
    # Custom fields that define their own mapping take priority.
    if hasattr(field_obj, "_jsonschema_type_mapping"):
        return field_obj._jsonschema_type_mapping()
    if "_jsonschema_type_mapping" in field_obj.metadata:
        return field_obj.metadata["_jsonschema_type_mapping"]

    # Nested schema
    if isinstance(field_obj, fields.Nested):
        nested_schema = field_obj.nested
        if isinstance(nested_schema, type) and issubclass(nested_schema, Schema):
            nested_schema = nested_schema()
        if isinstance(nested_schema, Schema):
            return _schema_to_jsonschema(nested_schema)
        return {"type": "object"}

    # Standard fields
    schema = _resolve_json_type(type(field_obj))
    schema["title"] = field_obj.attribute or field_obj.name or ""

    if field_obj.dump_only:
        schema["readOnly"] = True

    if field_obj.default is not missing and not callable(field_obj.default):
        schema["default"] = field_obj.default

    if field_obj.allow_none:
        prev = schema.get("type", "string")
        schema["type"] = [prev, "null"]

    # Copy metadata (description, parameter_metadata, etc.)
    metadata = field_obj.metadata.get("metadata", {})
    metadata.update(field_obj.metadata)
    for key, val in metadata.items():
        if key in ("metadata", "name"):
            continue
        schema[key] = val

    # List items
    if isinstance(field_obj, fields.List) and field_obj.inner is not None:
        schema["items"] = _field_to_jsonschema(field_obj.inner, parent_schema)

    # Dict values
    if isinstance(field_obj, fields.Dict) and field_obj.value_field is not None:
        schema["additionalProperties"] = _field_to_jsonschema(field_obj.value_field, parent_schema)

    # Validators
    for v in field_obj.validators:
        if isinstance(v, validate.OneOf):
            schema["enum"] = list(v.choices)
        elif isinstance(v, validate.Range):
            if v.min is not None:
                schema["minimum"] = v.min
            if v.max is not None:
                schema["maximum"] = v.max
        elif isinstance(v, validate.Length):
            # Use minItems/maxItems for arrays, minLength/maxLength for strings
            is_array = isinstance(field_obj, (fields.List, fields.Tuple))
            if v.min is not None:
                schema["minItems" if is_array else "minLength"] = v.min
            if v.max is not None:
                schema["maxItems" if is_array else "maxLength"] = v.max

    return schema


def _schema_to_jsonschema(schema_instance):
    """Convert a marshmallow Schema instance to a JSON Schema dict."""
    properties = OrderedDict()
    required = []

    for name, field_obj in schema_instance.fields.items():
        properties[name] = _field_to_jsonschema(field_obj, schema_instance)
        if field_obj.required:
            required.append(name)

    result = {
        "properties": properties,
        "type": "object",
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def marshmallow_schema_to_jsonschema_dict(schema_instance):
    """Convert a marshmallow Schema to a ``{"definitions": {...}}`` dict.

    This is a drop-in replacement for ``JSONSchema(props_ordered=True).dump(schema_instance)``.
    """
    name = type(schema_instance).__name__
    schema_dict = _schema_to_jsonschema(schema_instance)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {name: schema_dict},
        "$ref": f"#/definitions/{name}",
    }
