import os
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Union
from pprint import pprint

from torch.autograd.profiler_util import _format_memory, _format_time

import ludwig.modules.metric_modules  # noqa: F401
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, REPORT_JSON
from ludwig.modules.metric_registry import get_metric_classes, metric_feature_registry  # noqa: F401
from ludwig.utils.data_utils import load_json, flatten_dict


@dataclass
class MetricsSummary:
    """Summary of metrics from one experiment."""

    # Path containing the artifacts for the experiment.
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
class Diff:
    """Diffs for a metric."""

    # Name of the metric.
    name: str

    # Value of the metric in base experiment (the one we benchmark against).
    base_value: float

    # Value of the metric in the experimental experiment.
    experimental_value: float

    # experimental_value - base_value.
    diff: float

    # Percentage of change the metric with respect to base_value.
    diff_percentage: Union[float, str]

    def __post_init__(self):
        """Add human-readable string representations to the field."""

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
    """Store diffs for two experiments."""

    # Dataset the two experiments are being compared on.
    dataset_name: str

    # Name of the base experiment (the one we benchmark against).
    base_experiment_name: str

    # Name of the experimental experiment.
    experimental_experiment_name: str

    # Path under which all artifacts live on the local machine.
    local_directory: str

    # `MetricsSummary` of the base_experiment.
    base_summary: MetricsSummary

    # `MetricsSummary` of the experimental_experiment.
    experimental_summary: MetricsSummary

    # `List[Diff]` containing diffs for metric of the two experiments.
    metrics: List[Diff]

    def to_csv(self, path):
        csv_str = "{}, {}, {}, {}, {}, {}, {}\n"
        with open(path, "w") as f:
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
        print("Exported report to", path)

    def to_string(self):
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
    """Summary of resource usage metrics from one experiment."""

    # The tag with which the code block/function is labeled.
    code_block_tag: str

    # Dictionary that maps from metric name to their values.
    metric_to_values: Dict[str, Union[float, int]]

    # Names of metrics for the output feature.
    metric_names: set


@dataclass
class ResourceUsageDiff:
    """Store resource usage diffs for two experiments."""

    # The tag with which the code block/function is labeled.
    code_block_tag: str

    # Name of the base experiment (the one we benchmark against).
    base_experiment_name: str

    # Name of the experimental experiment.
    experimental_experiment_name: str

    # `List[Diff]` containing diffs for metric of the two experiments.
    metrics: List[Diff]

    def to_csv(self, path):
        csv_str = "{}, {}, {}, {}, {}, {}\n"
        with open(path, "w") as f:
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
        print("Exported report to", path)

    def to_string(self):
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


def build_metrics_diff(
        dataset_name: str, base_experiment_name: str, experimental_experiment_name: str, local_directory: str
) -> MetricsDiff:
    base_summary: MetricsSummary = build_metrics_summary(
        os.path.join(local_directory, dataset_name, base_experiment_name)
    )
    experimental_summary: MetricsSummary = build_metrics_summary(
        os.path.join(local_directory, dataset_name, experimental_experiment_name)
    )

    metrics_in_common = set(base_summary.metric_names).intersection(set(experimental_summary.metric_names))

    metrics: List[Diff] = [
        build_diff(name, base_summary.metric_to_values[name], experimental_summary.metric_to_values[name])
        for name in metrics_in_common
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
    report = {key: value for key, value in report.items() if isinstance(value, (int, float))}
    metric_names = set(report.keys())
    return ResourceUsageSummary(code_block_tag=code_block_tag, metric_to_values=report, metric_names=metric_names)


def build_resource_usage_diff_from_path(base_path, experimental_path):
    base_summary = build_resource_usage_summary(base_path)
    experimental_summary = build_resource_usage_summary(experimental_path)

    metrics_in_common = set(base_summary.metric_names).intersection(set(experimental_summary.metric_names))
    metrics: List[Diff] = [
        build_diff(name, base_summary.metric_to_values[name], experimental_summary.metric_to_values[name])
        for name in metrics_in_common
    ]
    diff = ResourceUsageDiff(
        code_block_tag=base_summary.code_block_tag,
        base_experiment_name=base_summary.code_block_tag,
        experimental_experiment_name=experimental_summary.code_block_tag,
        metrics=metrics,
    )
    return diff


def average_runs(path_to_runs_dir):
    """Return average metrics from code blocks/function that ran more than once.

    Metrics for code blocks/functions that were executed exactly once will be returned as is.
    """
    runs = [load_json(os.path.join(path_to_runs_dir, run)) for run in os.listdir(path_to_runs_dir)]
    # asserting that keys to each of the dictionaries are consistent throughout the runs.
    assert len(runs) == 1 or all(runs[i].keys() == runs[i + 1].keys() for i in range(len(runs) - 1))
    runs_average = {"num_runs": len(runs)}
    for key in runs[0].keys():
        if isinstance(runs[0][key], (int, float)):
            runs_average[key] = mean([run[key] for run in runs])
    return runs_average


def summarize_resource_usage(path, tags=None) -> List[ResourceUsageSummary]:
    summary = dict()
    # metric types: system_resource_usage, torch_ops_resource_usage.
    for metric_type in os.listdir(path):
        metric_type_path = os.path.join(path, metric_type)
        # code block tags correspond to the `tag` argument in ResourceUsageTracker.
        for code_block_tag in os.listdir(metric_type_path):
            if tags and code_block_tag not in tags:
                continue
            if code_block_tag not in summary:
                summary[code_block_tag] = {}
            run_path = os.path.join(metric_type_path, code_block_tag)
            runs_average = average_runs(run_path)
            summary[code_block_tag][metric_type] = runs_average

    summary_list = []
    for code_block_tag, metric_type_dicts in summary.items():
        merged_summary = {}
        for metrics in metric_type_dicts.values():
            assert "num_runs" in metrics
            assert "num_runs" not in merged_summary or metrics["num_runs"] == merged_summary["num_runs"]
            merged_summary.update(metrics)
        summary_list.append(ResourceUsageSummary(code_block_tag=code_block_tag, metric_to_values=merged_summary,
                                                 metric_names=set(merged_summary)))
    return summary_list


def build_resource_usage_diff(base_path, experimental_path):
    """Build and return a ResourceUsageDiff object to diff resource usage metrics between two experiments.

    base_path and experimental_path corresponds to the output_dir argument in
    ResourceUsageTracker for two different experiments
    """
    base_summary_list = summarize_resource_usage(base_path)
    experimental_summary_list = summarize_resource_usage(experimental_path)

    summaries_list = []
    for base_summary in base_summary_list:
        for experimental_summary in experimental_summary_list:
            if base_summary.code_block_tag == experimental_summary.code_block_tag:
                summaries_list.append((base_summary, experimental_summary))

    diffs = []
    for base_summary, experimental_summary in summaries_list:
        metrics_in_common = set(base_summary.metric_names).intersection(set(experimental_summary.metric_names))
        metrics: List[Diff] = [
            build_diff(name, base_summary.metric_to_values[name], experimental_summary.metric_to_values[name])
            for name in metrics_in_common
        ]
        diff = ResourceUsageDiff(
            code_block_tag=base_summary.code_block_tag,
            base_experiment_name=base_summary.code_block_tag,
            experimental_experiment_name=experimental_summary.code_block_tag,
            metrics=metrics,
        )
        diffs.append(diff)
    return diffs
