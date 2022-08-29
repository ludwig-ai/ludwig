from marshmallow import fields, ValidationError
from dataclasses import field

from ludwig.features.feature_registries import input_type_registry, output_type_registry
import ludwig.schema.utils as schema_utils


def get_defaults_conds():
    """Returns a JSON schema of conditionals to validate against encoder types for specific feature types."""
    conds = []
    for feat in input_type_registry.keys():
        input_feature_cls = input_type_registry.get(feat).get_schema_cls()
        output_feature_cls = output_type_registry.get(feat, None)
        if output_feature_cls:
            output_feature_cls = output_feature_cls.get_schema_cls()
        input_props = schema_utils.unload_jsonschema_from_marshmallow_class(input_feature_cls)["properties"]
        output_props = schema_utils.unload_jsonschema_from_marshmallow_class(output_feature_cls)["properties"]
        combined_props = {**input_props, **output_props}
        combined_props.pop("type")
        defaults_cond = schema_utils.create_cond(
            {"type": feat},
            combined_props,
        )
        conds.append(defaults_cond)
    return conds


def DefaultsDataclassField(feature_type: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a nested default config
    for a specific feature type.

    Returns: Initialized dataclass field that converts an untyped dict with params to a defaults config.
    """

    class DefaultMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid split config from the
        split_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                input_feature_class = input_type_registry[feature_type]
                output_feature_class = output_type_registry.get(feature_type, None)
                try:
                    input_schema = input_feature_class.get_schema_cls().Schema().load(value)
                    if output_feature_class:
                        output_schema = output_feature_class.get_schema_cls().Schema().load(value)
                        combined = input_schema + output_schema
                    else:
                        combined = input_schema
                    return combined
                except (TypeError, ValidationError) as error:
                    raise ValidationError(
                        f"Invalid params: {value}, see `{attr}` definition. Error: {error}"
                    )
            raise ValidationError(
                f"Invalid params: {value}"
            )

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(input_type_registry.data.keys()), "default": None},
                },
                "title": "defaults_options",
                "allOf": get_defaults_conds()
            }

    try:
        input_cls = input_type_registry[feature_type]
        output_cls = output_type_registry.get(feature_type, None)
        defaults = input_cls + output_cls
        load_default = defaults.Schema().load({"type": feature_type})
        dump_default = defaults.Schema().dump({"type": feature_type})

        return field(
            metadata={
                "marshmallow_field": DefaultMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported splitter type: {feature_type}. See split_registry. " f"Details: {e}")