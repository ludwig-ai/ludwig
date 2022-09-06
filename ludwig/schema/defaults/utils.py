from dataclasses import field

from marshmallow import fields, ValidationError

import ludwig.schema.utils as schema_utils
from ludwig.schema.features.utils import input_config_registry, output_config_registry


def DefaultsDataclassField(feature_type: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a nested default
    config for a specific feature type.

    Returns: Initialized dataclass field that converts an untyped dict with params to a defaults config.
    """

    class DefaultMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid defaults config from the feature_registry
        and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                input_feature_class = input_config_registry[feature_type]
                output_feature_class = output_config_registry.get(feature_type, None)
                try:
                    input_schema = input_feature_class.Schema().load(value)
                    if output_feature_class:
                        output_schema = output_feature_class.Schema().load(value)
                        combined = input_schema + output_schema
                    else:
                        combined = input_schema
                    return combined
                except (TypeError, ValidationError) as error:
                    raise ValidationError(f"Invalid params: {value}, see `{attr}` definition. Error: {error}")
            raise ValidationError(f"Invalid params: {value}")

        @staticmethod
        def _jsonschema_type_mapping():
            input_feature_cls = input_config_registry.get(feature_type)
            output_feature_cls = output_config_registry.get(feature_type, None)
            input_props = schema_utils.unload_jsonschema_from_marshmallow_class(input_feature_cls)["properties"]
            if output_feature_cls:
                output_props = schema_utils.unload_jsonschema_from_marshmallow_class(output_feature_cls)["properties"]
                combined_props = {**output_props, **input_props}
            else:
                combined_props = input_props
            return {
                "type": "object",
                "properties": combined_props,
                "title": "defaults_options",
            }

    try:
        input_cls = input_config_registry[feature_type]
        output_cls = output_config_registry.get(feature_type, None)
        dump_default = input_cls.Schema().dump({"type": feature_type})
        if output_cls:
            output_dump = output_cls.Schema().dump({"type": feature_type})
            dump_default = {**output_dump, **dump_default}

        load_default = input_cls.Schema().load({"type": feature_type})
        if output_cls:
            output_load = output_cls.Schema().load({"type": feature_type})
            for k in dump_default.keys():
                if getattr(load_default, k, -1) == -1:
                    setattr(load_default, k, getattr(output_load, k))
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
        raise ValidationError(f"Unsupported feature type: {feature_type}. See input_type_registry. " f"Details: {e}")
