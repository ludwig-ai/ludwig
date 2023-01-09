from dataclasses import field
from typing import List, Union

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.utils.registry import Registry

encoder_config_registry = Registry()


@DeveloperAPI
def register_encoder_config(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = encoder_config_registry.get(feature, {})
            feature_registry[name] = cls
            encoder_config_registry[feature] = feature_registry
        return cls

    return wrap


@DeveloperAPI
def get_encoder_cls(feature: str, name: str):
    return encoder_config_registry[feature][name]


@DeveloperAPI
def get_encoder_classes(feature: str):
    return encoder_config_registry[feature]


@DeveloperAPI
def get_encoder_descriptions(feature_type: str):
    """
    This function returns a dictionary of encoder descriptions available at the type selection.
    The process works as follows - 1) Get a dictionary of valid encoders from the encoder config registry,
    but inverse the key/value pairs since we need to index `valid_encoders` later with an altered version
    of the encoder config class name. 2) Loop through Encoder Metadata entries, if a metadata entry has an
    encoder name that matches a valid encoder, add the description metadata to the output dictionary.

    NOTE: We need to check if the metadata entry is in the valid encoder key name instead of checking for a
    match because sometimes there is a 1:many match on encoder metadata to encoder config class names. For
    example, the encoder metadata entry for "PassthroughEncoder" has a class name of "BinaryPassthroughEncoderConfig"

    :param feature_type: The feature type to get encoder descriptions for
    :return: A dictionary of encoder descriptions
    """
    output = {}

    # Get valid encoder dictionary with following structure:
    #       key - name of encoder config class altered to match metadata class names,
    #       value - registered name of encoder
    valid_encoders = {
        class_name.__name__.replace("Config", ""): registered_name
        for registered_name, class_name
        in get_encoder_classes(feature_type).items()
    }

    # Get encoder metadata entries for the valid encoders
    for k, v in ENCODER_METADATA.items():
        for name in valid_encoders:
            # Check if the metadata class name is in the valid encoder key name since the metadata class names sometimes
            # map one to many (i.e. BinaryPassthroughEncoder is PassthroughEncoder in metadata)
            if k in name and not isinstance(v, ParameterMetadata):
                # Set key to class name to index valid encoders correctly
                k = name
                output[valid_encoders[k]] = convert_metadata_to_json(v[TYPE])
    return output


@DeveloperAPI
def get_encoder_conds(feature_type: str):
    """Returns a JSON schema of conditionals to validate against encoder types for specific feature types."""
    conds = []
    for encoder in get_encoder_classes(feature_type):
        encoder_cls = get_encoder_cls(feature_type, encoder)
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(encoder_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        encoder_cond = schema_utils.create_cond(
            {"type": encoder},
            other_props,
        )
        conds.append(encoder_cond)
    return conds


@DeveloperAPI
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
                    enc = get_encoder_cls(feature_type, value[TYPE])
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
                    "type": {
                        "type": "string",
                        "enum": encoder_classes,
                        "enum_descriptions": get_encoder_descriptions(feature_type),
                        "default": default,
                    },
                },
                "title": "encoder_options",
                "allOf": get_encoder_conds(feature_type),
            }

    try:
        encoder = get_encoder_cls(feature_type, default)
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
