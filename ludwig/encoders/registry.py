from typing import Dict, List, Type, Union

from ludwig.encoders.base import Encoder
from ludwig.utils.registry import DEFAULT_KEYS, Registry

encoder_registry = Registry()

sequence_encoder_registry = Registry()


def register_sequence_encoder(name: str):
    def wrap(cls):
        sequence_encoder_registry[name] = cls
        return cls

    return wrap


def register_encoder(name: str, features: Union[str, List[str]], default=False):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = encoder_registry.get(feature, {})
            feature_registry[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    feature_registry[key] = cls
            encoder_registry[feature] = feature_registry
        return cls

    return wrap


def get_encoder_cls(feature: str, name: str) -> Type[Encoder]:
    return encoder_registry[feature][name]


def get_encoder_classes(feature: str) -> Dict[str, Type[Encoder]]:
    return encoder_registry[feature]
