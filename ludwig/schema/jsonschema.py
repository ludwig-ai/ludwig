"""JSON Schema generation for Ludwig config classes.

Uses pydantic's model_json_schema() under the hood, replacing the previous marshmallow-based converter.
"""


def marshmallow_schema_to_jsonschema_dict(schema_instance):
    """Backward-compatible JSON schema generation.

    Previously converted marshmallow schemas. Now uses pydantic's model_json_schema().
    The schema_instance can be either:
    - A pydantic model class (BaseMarshmallowConfig subclass)
    - A _SchemaAdapter instance
    - Legacy: called with a marshmallow Schema instance (raises helpful error)
    """
    from ludwig.schema.utils import _SchemaAdapter, BaseMarshmallowConfig

    # Handle _SchemaAdapter
    if isinstance(schema_instance, _SchemaAdapter):
        cls = schema_instance._cls
    elif isinstance(schema_instance, type) and issubclass(schema_instance, BaseMarshmallowConfig):
        cls = schema_instance
    elif isinstance(schema_instance, BaseMarshmallowConfig):
        cls = type(schema_instance)
    else:
        raise TypeError(
            f"Expected a Ludwig config class or schema adapter, got {type(schema_instance)}. "
            "Marshmallow schemas are no longer supported. Use pydantic BaseModel subclasses."
        )

    schema_dict = cls.model_json_schema()
    name = cls.__name__

    # Wrap in definitions format for backward compat
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {name: schema_dict},
        "$ref": f"#/definitions/{name}",
    }
