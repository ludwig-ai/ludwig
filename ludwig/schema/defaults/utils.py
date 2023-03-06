from dataclasses import field

from marshmallow import fields, ValidationError

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema.features.utils import ecd_defaults_config_registry
from ludwig.utils.registry import Registry


@DeveloperAPI
def DefaultsDataclassField(feature_type: str, defaults_registry: Registry = ecd_defaults_config_registry):
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
                defaults_class = defaults_registry[feature_type]
                try:
                    return defaults_class.Schema().load(value)
                except (TypeError, ValidationError) as error:
                    raise ValidationError(f"Invalid params: {value}, see `{attr}` definition. Error: {error}")
            raise ValidationError(f"Invalid params: {value}")

        def _jsonschema_type_mapping(self):
            defaults_cls = defaults_registry[feature_type]
            props = schema_utils.unload_jsonschema_from_marshmallow_class(defaults_cls)["properties"]
            return {
                "type": "object",
                "properties": props,
                "additionalProperties": False,
                "title": "defaults_options",
            }

    try:
        defaults_cls = defaults_registry[feature_type]
        dump_default = defaults_cls.Schema().dump({})
        load_default = lambda: defaults_cls.Schema().load({})

        return field(
            metadata={
                "marshmallow_field": DefaultMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported feature type: {feature_type}. Allowed: {defaults_registry.keys()}. " f"Details: {e}"
        )
