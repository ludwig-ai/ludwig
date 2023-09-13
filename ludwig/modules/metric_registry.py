from typing import Dict, List, Literal, TYPE_CHECKING, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOGITS, MAXIMIZE, MINIMIZE, PREDICTIONS, PROBABILITIES, RESPONSE
from ludwig.utils.registry import Registry

if TYPE_CHECKING:
    from ludwig.modules.metric_modules import LudwigMetric


metric_feature_type_registry = Registry()
metric_registry = Registry()
metric_objective_registry = Registry()
metric_tensor_input_registry = Registry()


def register_metric(
    name: str,
    feature_types: Union[str, List[str]],
    objective: Literal[MINIMIZE, MAXIMIZE],
    output_feature_tensor_name: Literal[PREDICTIONS, PROBABILITIES, LOGITS],
):
    """Registers a metric class.

    Args:
        name: The name of the metric. Used in metric reporting and in the config.
        feature_types: The feature types that this metric can be used with.
        objective: The objective of the metric. Either MINIMIZE or MAXIMIZE.
        output_feature_tensor_name: Name of the tensor from output_feature::predictions() that should be used as input.
            For example: PREDICTIONS would be used for accuracy metrics while LOGITS would be used for loss metrics.
    """
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    def wrap(cls):
        for feature_type in feature_types:
            feature_registry = metric_feature_type_registry.get(feature_type, {})
            feature_registry[name] = cls
            metric_feature_type_registry[feature_type] = feature_registry
        metric_registry[name] = cls
        metric_objective_registry[name] = objective
        metric_tensor_input_registry[name] = output_feature_tensor_name
        return cls

    return wrap


def get_metric_classes(feature_type: str) -> Dict[str, "LudwigMetric"]:
    return metric_feature_type_registry[feature_type]


def get_metric_cls(feature_type: str, name: str) -> "LudwigMetric":
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


@DeveloperAPI
def get_metric_objective(metric_name: str) -> Literal[MINIMIZE, MAXIMIZE]:
    return metric_objective_registry[metric_name]


@DeveloperAPI
def get_metric_tensor_input(metric_name: str) -> Literal[PREDICTIONS, PROBABILITIES, LOGITS, RESPONSE]:
    return metric_tensor_input_registry[metric_name]
