from typing import List, Union

from ludwig.utils.registry import Registry

metric_feature_type_registry = Registry()
metric_registry = Registry()


def register_metric(name: str, feature_types: Union[str, List[str]]):
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    def wrap(cls):
        for feature_type in feature_types:
            feature_registry = metric_feature_type_registry.get(feature_type, {})
            feature_registry[name] = cls
            metric_feature_type_registry[feature_type] = feature_registry
        metric_registry[name] = cls
        return cls

    return wrap


def get_metric_classes(feature_type: str):
    return metric_feature_type_registry[feature_type]


def get_metric_cls(feature_type: str, name: str):
    return metric_feature_type_registry[feature_type][name]
