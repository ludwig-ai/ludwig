from dataclasses import Field
from typing import Dict, List, Optional, Type, TYPE_CHECKING, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD
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
def EncoderDataclassField(model_type: str, feature_type: str, default: str) -> Field:
    """Custom dataclass field that when used inside a dataclass will allow the user to specify an encoder config.

    Returns: Initialized dataclass field that converts an untyped dict with params to an encoder config.
    """
    encoder_registry = get_encoder_classes(model_type, feature_type)

    class EncoderSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(registry=encoder_registry, default_value=default)

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return encoder_registry[key]

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(encoder_registry.keys()), "default": default},
                },
                "title": "encoder_options",
                "allOf": get_encoder_conds(encoder_registry),
            }

    return EncoderSelection().get_default_field()
