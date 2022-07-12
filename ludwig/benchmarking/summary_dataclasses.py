import os
from dataclasses import dataclass
from typing import List

from globals import REPORT_JSON

from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.modules.metric_registry import get_metric_classes
from ludwig.utils.data_utils import load_json


@dataclass
class ExperimentSummary:
    """Summary of metrics from one experiment.

    experiment_local_directory: path containing the artifacts for the experiment.
    output_feature_type: LudwigModel output feature type.
    output_feature_name: LudwigModel output feature name.
    metric_to_values: dictionary that maps from metric name to their values.
    metric_names: names of metrics for the output feature.
    empty: True if unable to load metrics.
    """

    experiment_local_directory: str

    def __post_init__(self):
        self.errors: List[Exception] = []
        try:
            self.config = load_json(os.path.join(self.experiment_local_directory, MODEL_HYPERPARAMETERS_FILE_NAME))
            report = load_json(os.path.join(self.experiment_local_directory, REPORT_JSON))
        except Exception as error:
            self.errors.append(error)
            self.empty = True
            return

        performance_metrics = report["evaluate"]["performance_metrics"]
        self.output_feature_type: str = self.config["output_features"][0]["type"]
        self.output_feature_name: str = self.config["output_features"][0]["name"]
        metric_dict = performance_metrics[self.output_feature_name]
        full_metric_names = get_metric_classes(self.output_feature_type)
        self.metric_to_values: dict = {
            metric_name: metric_dict[metric_name] for metric_name in full_metric_names if metric_name in metric_dict
        }
        self.metric_names: set = set(self.metric_to_values.keys())
        self.empty = False


@dataclass
class MetricDiff:
    """Diffs for a metric.

    name: name of the metric.
    base_value: value of the metric in base experiment (the one we benchmark against).
    experimental_value: value of the metric in the experimental experiment.
    diff: experimental_value - base_value.
    diff_percentage: percentage of change the metric with respect to base_value.
    """

    name: str
    base_value: float
    experimental_value: float

    def __post_init__(self):
        self.diff = self.experimental_value - self.base_value
        self.diff_percentage = 100 * self.diff / self.base_value if self.base_value != 0 else None


@dataclass
class ExperimentsDiff:
    """Store diffs for two experiments.

    dataset_name: dataset the two experiments are being compared on.
    base_experiment_name: name of the base experiment (the one we benchmark against).
    experimental_experiment_name: name of the experimental experiment.
    local_directory: path under which all artifacts live on the local machine.
    base_summary: `ExperimentSummary` of the base_experiment.
    experimental_summary: `ExperimentSummary` of the experimental_experiment.
    metrics: `List[MetricDiff]` containing diffs for metric of the two experiments.
    empty: True if we're unable to load either of the `ExperimentSummary`.
    """

    dataset_name: str
    base_experiment_name: str
    experimental_experiment_name: str
    local_directory: str

    def __post_init__(self):
        self.base_summary: ExperimentSummary = ExperimentSummary(
            os.path.join(self.local_directory, self.dataset_name, self.base_experiment_name)
        )
        self.experimental_summary: ExperimentSummary = ExperimentSummary(
            os.path.join(self.local_directory, self.dataset_name, self.experimental_experiment_name)
        )

        if self.base_summary.empty or self.experimental_summary.empty:
            self.empty = True
            return

        shared_metrics = set(self.base_summary.metric_names).intersection(set(self.experimental_summary.metric_names))
        self.metrics: List[MetricDiff] = [
            MetricDiff(name, self.base_summary.metric_to_values[name], self.experimental_summary.metric_to_values[name])
            for name in shared_metrics
        ]
        self.empty = False
