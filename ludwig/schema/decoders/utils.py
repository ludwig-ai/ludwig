from dataclasses import Field
from typing import List, Type, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
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
def DecoderDataclassField(feature_type: str, default: str) -> Field:
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a decoder config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a decoder config.
    """
    decoder_registry = get_decoder_classes(feature_type)

    class DecoderSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(registry=decoder_registry, default_value=default)

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return get_decoder_cls(feature_type, key)

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(decoder_registry.keys()), "default": default},
                },
                "title": "decoder_options",
                "allOf": get_decoder_conds(feature_type),
            }

    return DecoderSelection().get_default_field()
