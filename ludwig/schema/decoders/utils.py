from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.constants import TYPE
from ludwig.decoders.registry import get_decoder_classes, get_decoder_cls
from ludwig.schema import utils as schema_utils


def get_decoder_conds(feature_type: str):
    """Returns a JSON schema of conditionals to validate against decoder types for specific feature types."""
    conds = []
    for decoder in get_decoder_classes(feature_type):
        decoder_cls = get_decoder_cls(feature_type, decoder).get_schema_cls()
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(decoder_cls)["properties"]
        other_props.pop("type")
        decoder_cond = schema_utils.create_cond(
            {"type": decoder},
            other_props,
        )
        conds.append(decoder_cond)
    return conds


def DecoderDataclassField(feature_type: str, default: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a decoder config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a decoder config.
    """

    class DecoderMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid decoder config from the decoder_registry
        and creates a corresponding `oneOf` JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if TYPE in value and value[TYPE] in get_decoder_classes(feature_type):
                    enc = get_decoder_cls(feature_type, default).get_schema_cls()
                    try:
                        return enc.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid decoder params: {value}, see `{enc}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for decoder: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            decoder_classes = list(get_decoder_classes(feature_type).keys())

            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": decoder_classes, "default": default},
                },
                "title": "decoder_options",
                "allOf": get_decoder_conds(feature_type),
                "required": ["type"],
            }

    try:
        decoder = get_decoder_cls(feature_type, default).get_schema_cls()
        load_default = decoder.Schema().load({"type": default})
        dump_default = decoder.Schema().dump({"type": default})

        return field(
            metadata={
                "marshmallow_field": DecoderMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported decoder type: {default}. See decoder_registry. " f"Details: {e}")
