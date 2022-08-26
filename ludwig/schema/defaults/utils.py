from marshmallow import fields, ValidationError
from dataclasses import field

from ludwig.features.feature_registries import input_type_registry, output_type_registry


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
                output_feature_class = output_type_registry[feature_type]
                try:
                    return split_class.get_schema_cls().Schema().load(value)
                except (TypeError, ValidationError) as error:
                    raise ValidationError(
                        f"Invalid split params: {value}, see `{split_class}` definition. Error: {error}"
                    )
            raise ValidationError(
                f"Invalid params for splitter: {value}, expected dict with at least a valid `type` attribute."
            )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(split_config_registry.data.keys()), "default": default},
                },
                "title": "split_options",
                "allOf": get_split_conds()
            }

    try:
        splitter = split_config_registry.data[default]
        load_default = splitter.Schema().load({"type": default})
        dump_default = splitter.Schema().dump({"type": default})

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
        raise ValidationError(f"Unsupported splitter type: {default}. See split_registry. " f"Details: {e}")