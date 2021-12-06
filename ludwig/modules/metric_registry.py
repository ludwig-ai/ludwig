from typing import List, Union

from ludwig.utils.registry import Registry

metric_feature_registry = Registry()
metric_registry = Registry()


def register_metric(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = metric_feature_registry.get(feature, {})
            feature_registry[name] = cls
            metric_feature_registry[feature] = feature_registry
        metric_registry[name] = cls
        return cls

    return wrap


def get_metric_classes(feature: str):
    return metric_feature_registry[feature]


def get_metric_cls(feature: str, name: str):
    return metric_feature_registry[feature][name]
