import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, REPORT_JSON
from ludwig.modules.metric_registry import get_metric_classes
from ludwig.utils.data_utils import load_json


@dataclass
class ExperimentSummary:
    """Summary of metrics from one experiment."""

    # path containing the artifacts for the experiment.
    experiment_local_directory: str

    # Full Ludwig config.
    config: Dict[str, Any]

    # LudwigModel output feature type.
    output_feature_type: str

    # LudwigModel output feature name.
    output_feature_name: str

    # Dictionary that maps from metric name to their values.
    metric_to_values: Dict[str, Union[float, int]]

    # Names of metrics for the output feature.
    metric_names: set


@dataclass
class MetricDiff:
    """Diffs for a metric."""

    # Name of the metric.
    name: str

    # Value of the metric in base experiment (the one we benchmark against).
    base_value: float

    # Value of the metric in the experimental experiment.
    experimental_value: float

    # Experimental_value - base_value.
    diff: float

    # Percentage of change the metric with respect to base_value.
    diff_percentage: float


@dataclass
class ExperimentsDiff:
    """Store diffs for two experiments."""

    # Dataset the two experiments are being compared on.
    dataset_name: str

    # Name of the base experiment (the one we benchmark against).
    base_experiment_name: str

    # Name of the experimental experiment.
    experimental_experiment_name: str

    # Path under which all artifacts live on the local machine.
    local_directory: str

    # Experiment summary of the base experiment.
    base_summary: ExperimentSummary

    # Experiment summary of the experimental experiment.
    experimental_summary: ExperimentSummary

    # List of metric diffs for a metric of the two experiments.
    metrics: List[MetricDiff]


def build_experiment_summary(experiment_local_directory: str) -> ExperimentSummary:
    config = load_json(os.path.join(experiment_local_directory, MODEL_HYPERPARAMETERS_FILE_NAME))
    report = load_json(os.path.join(experiment_local_directory, REPORT_JSON))
    performance_metrics = report["evaluate"]
    output_feature_type: str = config["output_features"][0]["type"]
    output_feature_name: str = config["output_features"][0]["name"]
    metric_dict = performance_metrics[output_feature_name]
    full_metric_names = get_metric_classes(output_feature_type)
    metric_to_values: dict = {
        metric_name: metric_dict[metric_name] for metric_name in full_metric_names if metric_name in metric_dict
    }
    metric_names: set = set(metric_to_values.keys())

    return ExperimentSummary(
        experiment_local_directory=experiment_local_directory,
        config=config,
        output_feature_name=output_feature_name,
        output_feature_type=output_feature_type,
        metric_to_values=metric_to_values,
        metric_names=metric_names,
    )


def build_metric_diff(name: str, base_value: float, experimental_value: float) -> MetricDiff:
    diff = experimental_value - base_value
    diff_percentage = 100 * diff / base_value if base_value != 0 else None

    return MetricDiff(
        name=name,
        base_value=base_value,
        experimental_value=experimental_value,
        diff=diff,
        diff_percentage=diff_percentage,
    )


def build_experiments_diff(
    dataset_name: str, base_experiment_name: str, experimental_experiment_name: str, local_directory: str
) -> ExperimentsDiff:
    base_summary: ExperimentSummary = build_experiment_summary(
        os.path.join(local_directory, dataset_name, base_experiment_name)
    )
    experimental_summary: ExperimentSummary = build_experiment_summary(
        os.path.join(local_directory, dataset_name, experimental_experiment_name)
    )

    shared_metrics = set(base_summary.metric_names).intersection(set(experimental_summary.metric_names))

    metrics: List[MetricDiff] = [
        build_metric_diff(name, base_summary.metric_to_values[name], experimental_summary.metric_to_values[name])
        for name in shared_metrics
    ]

    return ExperimentsDiff(
        dataset_name=dataset_name,
        base_experiment_name=base_experiment_name,
        experimental_experiment_name=experimental_experiment_name,
        local_directory=local_directory,
        base_summary=base_summary,
        experimental_summary=experimental_summary,
        metrics=metrics,
    )
