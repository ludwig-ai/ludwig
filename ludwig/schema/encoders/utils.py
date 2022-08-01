from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.constants import TYPE
from ludwig.encoders.registry import get_encoder_classes, get_encoder_cls
from ludwig.schema import utils as schema_utils


def get_encoder_conds(feature_type: str):
    """Returns a JSON schema of conditionals to validate against encoder types for specific feature types."""
    conds = []
    for encoder in get_encoder_classes(feature_type):
        encoder_cls = get_encoder_cls(feature_type, encoder).get_schema_cls()
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(encoder_cls)["properties"]
        other_props.pop("type")
        encoder_cond = schema_utils.create_cond(
            {"type": encoder},
            other_props,
        )
        conds.append(encoder_cond)
    return conds


def EncoderDataclassField(feature_type: str, default: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify an encoder config.

    Returns: Initialized dataclass field that converts an untyped dict with params to an encoder config.
    """

    class EncoderMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid encoder config from the encoder_registry
        and creates a corresponding `oneOf` JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if TYPE in value and value[TYPE] in get_encoder_classes(feature_type):
                    enc = get_encoder_cls(feature_type, default).get_schema_cls()
                    try:
                        return enc.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid encoder params: {value}, see `{enc}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for encoder: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            encoder_classes = list(get_encoder_classes(feature_type).keys())

            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": encoder_classes, "default": default},
                },
                "title": "encoder_options",
                "allOf": get_encoder_conds(feature_type),
                "required": ["type"],
            }

    try:
        encoder = get_encoder_cls(feature_type, default).get_schema_cls()
        load_default = encoder.Schema().load({"type": default})
        dump_default = encoder.Schema().dump({"type": default})

        return field(
            metadata={
                "marshmallow_field": EncoderMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported encoder type: {default}. See encoder_registry. " f"Details: {e}")
