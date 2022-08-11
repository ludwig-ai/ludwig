import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Union

from torch.autograd.profiler_util import _format_memory, _format_time

import ludwig.modules.metric_modules
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, REPORT_JSON
from ludwig.modules.metric_registry import get_metric_classes, metric_feature_registry
from ludwig.utils.data_utils import load_json


@dataclass
class MetricsSummary:
    """Summary of metrics from one experiment.

    experiment_local_directory: path containing the artifacts for the experiment.
    output_feature_type: LudwigModel output feature type.
    output_feature_name: LudwigModel output feature name.
    metric_to_values: dictionary that maps from metric name to their values.
    metric_names: names of metrics for the output feature.
    """

    experiment_local_directory: str
    config: Dict[str, Any]
    output_feature_type: str
    output_feature_name: str
    metric_to_values: Dict[str, Union[float, int]]
    metric_names: set


@dataclass
class Diff:
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
    diff: float
    diff_percentage: Union[float, str]

    def __post_init__(self):
        if "memory" in self.name:
            self.base_value_str = _format_memory(self.base_value)
            self.experimental_value_str = _format_memory(self.experimental_value)
            self.diff_str = _format_memory(self.diff)
        elif "time" in self.name:
            self.base_value_str = _format_time(self.base_value)
            self.experimental_value_str = _format_time(self.experimental_value)
            self.diff_str = _format_time(self.diff)
        else:
            self.base_value_str = str(self.base_value)
            self.experimental_value_str = str(self.experimental_value)
            self.diff_str = str(self.diff)


@dataclass
class MetricsDiff:
    """Store diffs for two experiments.

    dataset_name: dataset the two experiments are being compared on.
    base_experiment_name: name of the base experiment (the one we benchmark against).
    experimental_experiment_name: name of the experimental experiment.
    local_directory: path under which all artifacts live on the local machine.
    base_summary: `ExperimentSummary` of the base_experiment.
    experimental_summary: `ExperimentSummary` of the experimental_experiment.
    metrics: `List[MetricDiff]` containing diffs for metric of the two experiments.
    """

    dataset_name: str
    base_experiment_name: str
    experimental_experiment_name: str
    local_directory: str
    base_summary: MetricsSummary
    experimental_summary: MetricsSummary
    metrics: List[Diff]

    def to_csv(self):
        csv_str = "{}, {}, {}, {}, {}, {}, {}\n"
        file_name = f"report_{self.dataset_name}_{self.base_experiment_name}_{self.experimental_experiment_name}.csv"
        with open(file_name, "w") as f:
            f.write(
                csv_str.format(
                    "Dataset Name",
                    "Output Feature Name",
                    "Metric Name",
                    self.base_experiment_name,
                    self.experimental_experiment_name,
                    "Diff",
                    "Diff Percentage",
                )
            )
            for metric in sorted(self.metrics, key=lambda m: m.name):
                output_feature_name = self.base_summary.output_feature_name
                metric_name = metric.name
                experiment1_val = round(metric.base_value, 3)
                experiment2_val = round(metric.experimental_value, 3)
                diff = round(metric.diff, 3)
                diff_percentage = metric.diff_percentage
                if isinstance(diff_percentage, float):
                    diff_percentage = round(metric.diff_percentage, 3)

                f.write(
                    csv_str.format(
                        self.dataset_name,
                        output_feature_name,
                        metric_name,
                        experiment1_val,
                        experiment2_val,
                        diff,
                        diff_percentage,
                    )
                )

        print("Exported report to", file_name)

    def __str__(self):
        ret = []
        spacing_str = "{:<20} {:<23} {:<13} {:<13} {:<13} {:<5}"
        ret.append(f"Metrics for dataset: {self.dataset_name}\n")
        ret.append(
            spacing_str.format(
                "Output Feature Name",
                "Metric Name",
                self.base_experiment_name,
                self.experimental_experiment_name,
                "Diff",
                "Diff Percentage",
            )
        )

        for metric in sorted(self.metrics, key=lambda m: m.name):
            output_feature_name = self.base_summary.output_feature_name
            metric_name = metric.name
            experiment1_val = round(metric.base_value, 3)
            experiment2_val = round(metric.experimental_value, 3)
            diff = round(metric.diff, 3)
            diff_percentage = metric.diff_percentage
            if isinstance(diff_percentage, float):
                diff_percentage = round(metric.diff_percentage, 3)
            ret.append(
                spacing_str.format(
                    output_feature_name,
                    metric_name,
                    experiment1_val,
                    experiment2_val,
                    diff,
                    diff_percentage,
                )
            )
        return "\n".join(ret)


@dataclass
class ResourceUsageSummary:
    """Summary of metrics from one experiment.

    experiment_local_directory: path containing the artifacts for the experiment.
    output_feature_type: LudwigModel output feature type.
    output_feature_name: LudwigModel output feature name.
    metric_to_values: dictionary that maps from metric name to their values.
    metric_names: names of metrics for the output feature.
    """

    path: str
    code_block_tag: str
    metric_to_values: Dict[str, Union[float, int]]
    metric_names: set


@dataclass
class ResourceUsageDiff:
    """Store diffs for two experiments.

    dataset_name: dataset the two experiments are being compared on.
    base_experiment_name: name of the base experiment (the one we benchmark against).
    experimental_experiment_name: name of the experimental experiment.
    local_directory: path under which all artifacts live on the local machine.
    base_summary: `ExperimentSummary` of the base_experiment.
    experimental_summary: `ExperimentSummary` of the experimental_experiment.
    metrics: `List[MetricDiff]` containing diffs for metric of the two experiments.
    """

    code_block_tag: str
    base_experiment_name: str
    experimental_experiment_name: str
    metrics: List[Diff]

    def to_csv(self):
        csv_str = "{}, {}, {}, {}, {}, {}\n"
        file_name = f"report_{self.code_block_tag}_{self.base_experiment_name}_{self.experimental_experiment_name}.csv"
        with open(file_name, "w") as f:
            f.write(
                csv_str.format(
                    "Code Block Tag",
                    "Metric Name",
                    self.base_experiment_name,
                    self.experimental_experiment_name,
                    "Diff",
                    "Diff Percentage",
                )
            )
            for metric in sorted(self.metrics, key=lambda m: m.name):
                diff_percentage = metric.diff_percentage
                if isinstance(metric.diff_percentage, float):
                    diff_percentage = round(metric.diff_percentage, 3)
                f.write(
                    csv_str.format(
                        self.code_block_tag,
                        metric.name,
                        metric.base_value_str,
                        metric.experimental_value_str,
                        metric.diff_str,
                        diff_percentage,
                    )
                )
        print("Exported report to", file_name)

    def __str__(self):
        ret = []
        spacing_str = "{:<30} {:<20} {:<20} {:<20} {:<5}"
        ret.append(f"\nResource usage for: {self.code_block_tag}")
        ret.append(
            spacing_str.format(
                "Metric Name",
                self.base_experiment_name,
                self.experimental_experiment_name,
                "Diff",
                "Diff Percentage",
            )
        )

        for metric in sorted(self.metrics, key=lambda m: m.name):
            diff_percentage = metric.diff_percentage
            if isinstance(metric.diff_percentage, float):
                diff_percentage = round(metric.diff_percentage, 3)
            ret.append(
                spacing_str.format(
                    metric.name,
                    metric.base_value_str,
                    metric.experimental_value_str,
                    metric.diff_str,
                    diff_percentage,
                )
            )
        return "\n".join(ret)


def build_metrics_summary(experiment_local_directory: str) -> MetricsSummary:
    config = load_json(os.path.join(experiment_local_directory, MODEL_HYPERPARAMETERS_FILE_NAME))
    report = load_json(os.path.join(experiment_local_directory, REPORT_JSON))
    performance_metrics = report["evaluate"]["performance_metrics"]
    output_feature_type: str = config["output_features"][0]["type"]
    output_feature_name: str = config["output_features"][0]["name"]
    metric_dict = performance_metrics[output_feature_name]
    full_metric_names = get_metric_classes(output_feature_type)
    metric_to_values: dict = {
        metric_name: metric_dict[metric_name] for metric_name in full_metric_names if metric_name in metric_dict
    }
    metric_names: set = set(metric_to_values.keys())

    return MetricsSummary(
        experiment_local_directory=experiment_local_directory,
        config=config,
        output_feature_name=output_feature_name,
        output_feature_type=output_feature_type,
        metric_to_values=metric_to_values,
        metric_names=metric_names,
    )


def build_diff(name: str, base_value: float, experimental_value: float) -> Diff:
    diff = experimental_value - base_value
    diff_percentage = 100 * diff / base_value if base_value != 0 else "inf"

    return Diff(
        name=name,
        base_value=base_value,
        experimental_value=experimental_value,
        diff=diff,
        diff_percentage=diff_percentage,
    )


def build_metrics_diff(
    dataset_name: str, base_experiment_name: str, experimental_experiment_name: str, local_directory: str
) -> MetricsDiff:
    base_summary: MetricsSummary = build_metrics_summary(
        os.path.join(local_directory, dataset_name, base_experiment_name)
    )
    experimental_summary: MetricsSummary = build_metrics_summary(
        os.path.join(local_directory, dataset_name, experimental_experiment_name)
    )

    shared_metrics = set(base_summary.metric_names).intersection(set(experimental_summary.metric_names))

    metrics: List[Diff] = [
        build_diff(name, base_summary.metric_to_values[name], experimental_summary.metric_to_values[name])
        for name in shared_metrics
    ]

    return MetricsDiff(
        dataset_name=dataset_name,
        base_experiment_name=base_experiment_name,
        experimental_experiment_name=experimental_experiment_name,
        local_directory=local_directory,
        base_summary=base_summary,
        experimental_summary=experimental_summary,
        metrics=metrics,
    )


def build_resource_usage_summary(path):
    report = load_json(path)
    code_block_tag = report["code_block_tag"]
    # num_runs = report["num_runs"]
    runs = report["runs"]

    def average_runs(runs):
        average_run = {"num_runs": len(runs)}
        for metric in runs[0].keys():
            average_run[metric] = mean([run[metric] for run in runs])
        return average_run

    average_run = average_runs(runs)
    metric_names = set(average_run.keys())
    return ResourceUsageSummary(
        path=path, code_block_tag=code_block_tag, metric_to_values=average_run, metric_names=metric_names
    )


def build_resource_usage_diff(
    dataset_name: str, base_experiment_name: str, experimental_experiment_name: str, local_directory: str
):
    base_dir = os.path.join(local_directory, dataset_name, base_experiment_name)
    experimental_dir = os.path.join(local_directory, dataset_name, experimental_experiment_name)
    return build_resource_usage_diff_from_path(
        base_dir, experimental_dir, base_experiment_name, experimental_experiment_name
    )


def build_resource_usage_diff_from_path(
    base_dir, experimental_dir, base_experiment_name="", experimental_experiment_name=""
):
    base_experiment_reports = set(os.listdir(base_dir))
    experimental_experiment_reports = set(os.listdir(experimental_dir))
    shared_reports = base_experiment_reports.intersection(experimental_experiment_reports)

    diffs = []
    for report in shared_reports:
        base_path = os.path.join(base_dir, report)
        experimental_path = os.path.join(experimental_dir, report)
        base_summary = build_resource_usage_summary(base_path)
        experimental_summary = build_resource_usage_summary(experimental_path)

        shared_metrics = set(base_summary.metric_names).intersection(set(experimental_summary.metric_names))
        metrics: List[Diff] = [
            build_diff(name, base_summary.metric_to_values[name], experimental_summary.metric_to_values[name])
            for name in shared_metrics
        ]
        diff = ResourceUsageDiff(
            code_block_tag=base_summary.code_block_tag,
            base_experiment_name=base_experiment_name,
            experimental_experiment_name=experimental_experiment_name,
            metrics=metrics,
        )
        diffs.append(diff)
    return diffs
