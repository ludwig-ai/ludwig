from typing import List, Union, Dict

from ludwig.api_annotations import DeveloperAPI
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


@DeveloperAPI
def get_metric_feature_type_registry() -> Registry:
    return metric_feature_type_registry


@DeveloperAPI
def get_metric_registry() -> Registry:
    return metric_registry


@DeveloperAPI
def get_metric(metric_name: str) -> "LudwigMetric":  # noqa
    return get_metric_registry()[metric_name]


@DeveloperAPI
def get_metrics_for_type(feature_type: str) -> Dict[str, "LudwigMetric"]:  # noqa
    return get_metric_feature_type_registry()[feature_type]


@DeveloperAPI
def get_metric_names_for_type(feature_type: str) -> List[str]:
    return sorted(list(get_metric_feature_type_registry()[feature_type].keys()))
