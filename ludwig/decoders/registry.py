from typing import Dict, List, Type, Union

from ludwig.decoders.base import Decoder
from ludwig.utils.registry import DEFAULT_KEYS, Registry

decoder_registry = Registry()


def register_decoder(name: str, features: Union[str, List[str]], default=False):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = decoder_registry.get(feature, {})
            feature_registry[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    feature_registry[key] = cls
            decoder_registry[feature] = feature_registry
        return cls

    return wrap


def get_decoder_cls(feature: str, name: str) -> Type[Decoder]:
    return decoder_registry[feature][name]


def get_decoder_classes(feature: str) -> Dict[str, Type[Decoder]]:
    return decoder_registry[feature]
