from typing import Dict, List, Type, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.decoders.base import Decoder
from ludwig.utils.registry import Registry

_decoder_registry = Registry()


@DeveloperAPI
def get_decoder_registry() -> Registry:
    return _decoder_registry


@DeveloperAPI
def register_decoder(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = get_decoder_registry().get(feature, {})
            feature_registry[name] = cls
            get_decoder_registry()[feature] = feature_registry
        return cls

    return wrap


@DeveloperAPI
def get_decoder_cls(feature: str, name: str) -> Type[Decoder]:
    return get_decoder_registry()[feature][name]


@DeveloperAPI
def get_decoder_classes(feature: str) -> Dict[str, Type[Decoder]]:
    return get_decoder_registry()[feature]
