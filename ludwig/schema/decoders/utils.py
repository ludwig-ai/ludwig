from dataclasses import field
from typing import List, Union

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import DECODER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.utils.registry import Registry

decoder_config_registry = Registry()


@DeveloperAPI
def register_decoder_config(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = decoder_config_registry.get(feature, {})
            feature_registry[name] = cls
            decoder_config_registry[feature] = feature_registry
        return cls

    return wrap


@DeveloperAPI
def get_decoder_cls(feature: str, name: str):
    return decoder_config_registry[feature][name]


@DeveloperAPI
def get_decoder_classes(feature: str):
    return decoder_config_registry[feature]


@DeveloperAPI
def get_decoder_descriptions(feature_type: str):
    """This function returns a dictionary of decoder descriptions available at the type selection.

    The process works as follows - 1) Get a dictionary of valid decoders from the decoder config registry,
    but inverse the key/value pairs since we need to index `valid_decoders` later with an altered version
    of the decoder config class name. 2) Loop through Decoder Metadata entries, if a metadata entry has a
    decoder name that matches a valid decoder, add the description metadata to the output dictionary.

    Args:
        feature_type (str): The feature type to get decoder descriptions for
    Returns:
        dict: A dictionary of decoder descriptions
    """
    output = {}
    valid_decoders = {
        cls.module_name(): registered_name for registered_name, cls in get_decoder_classes(feature_type).items()
    }

    for k, v in DECODER_METADATA.items():
        if k in valid_decoders.keys():
            output[valid_decoders[k]] = convert_metadata_to_json(v[TYPE])

    return output


@DeveloperAPI
def get_decoder_conds(feature_type: str):
    """Returns a JSON schema of conditionals to validate against decoder types for specific feature types."""
    conds = []
    for decoder in get_decoder_classes(feature_type):
        decoder_cls = get_decoder_cls(feature_type, decoder)
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(decoder_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        decoder_cond = schema_utils.create_cond(
            {"type": decoder},
            other_props,
        )
        conds.append(decoder_cond)
    return conds


@DeveloperAPI
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
                    dec = get_decoder_cls(feature_type, value[TYPE])
                    try:
                        return dec.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid decoder params: {value}, see `{dec}` definition. Error: {error}"
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
                    "type": {
                        "type": "string",
                        "enum": decoder_classes,
                        "enumDescriptions": get_decoder_descriptions(feature_type),
                        "default": default,
                    },
                },
                "title": "decoder_options",
                "allOf": get_decoder_conds(feature_type),
            }

    try:
        decoder = get_decoder_cls(feature_type, default)
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
