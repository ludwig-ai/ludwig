from dataclasses import field
from typing import Dict, List, Optional, Type, Union, TYPE_CHECKING

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD, TYPE
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

if TYPE_CHECKING:
    from ludwig.schema.encoders.base import BaseEncoderConfig

encoder_config_registry = Registry()


@DeveloperAPI
def register_encoder_config(name: str, features: Union[str, List[str]], model_types: Optional[List[str]] = None):
    if model_types is None:
        model_types = [MODEL_ECD]

    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for model_type in model_types:
            for feature in features:
                key = (model_type, feature)
                feature_registry = encoder_config_registry.get(key, {})
                feature_registry[name] = cls
                encoder_config_registry[key] = feature_registry
        return cls

    return wrap


@DeveloperAPI
def get_encoder_cls(model_type: str, feature: str, name: str):
    return encoder_config_registry[(model_type, feature)][name]


@DeveloperAPI
def get_encoder_classes(model_type: str, feature: str) -> Dict[str, Type["BaseEncoderConfig"]]:
    return encoder_config_registry[(model_type, feature)]


@DeveloperAPI
def get_encoder_conds(encoder_classes: Dict[str, Type["BaseEncoderConfig"]]):
    """Returns a JSON schema of conditionals to validate against encoder types for specific feature types."""
    conds = []
    for encoder_type, encoder_cls in encoder_classes.items():
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(encoder_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        encoder_cond = schema_utils.create_cond(
            {"type": encoder_type},
            other_props,
        )
        conds.append(encoder_cond)
    return conds


@DeveloperAPI
def EncoderDataclassField(model_type: str, feature_type: str, default: str):
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
                value_type = value.get(TYPE)
                if value_type:
                    encoder_classes = get_encoder_classes(model_type, feature_type)
                    print(value_type, list(encoder_classes.keys()))
                    if value_type in encoder_classes:
                        enc = encoder_classes[value_type]
                        try:
                            return enc.Schema().load(value)
                        except (TypeError, ValidationError) as e:
                            raise ValidationError(f"Invalid encoder params: {value}, see `{enc}` definition") from e
                    else:
                        # TODO(travis): why is Marshmallow swallowing this error?
                        raise ValidationError(
                            f"Invalid encoder type (model_type={model_type}, feature={feature_type}): "
                            f"'{value_type}', expected one of: {list(encoder_classes.keys())}"
                        )
                raise ValidationError(
                    f"Invalid params for encoder: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            encoder_classes = get_encoder_classes(model_type, feature_type)
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(encoder_classes.keys()), "default": default},
                },
                "title": "encoder_options",
                "allOf": get_encoder_conds(encoder_classes),
            }

    try:
        encoder = get_encoder_cls(model_type, feature_type, default)
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
