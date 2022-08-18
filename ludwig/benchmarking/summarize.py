import logging
import os.path
from typing import List, Tuple

from ludwig.benchmarking.summary_dataclasses import (
    build_metrics_diff,
    build_resource_usage_diff,
    MetricsDiff,
    ResourceUsageDiff,
)
from ludwig.benchmarking.utils import download_artifacts


def summarize_metrics(
    bench_config_path: str, base_experiment: str, experimental_experiment: str, download_base_path: str
) -> Tuple[List[MetricsDiff], List[List[ResourceUsageDiff]]]:
    """Build metric and resource usage diffs from experiment artifacts.

    bench_config_path: bench config file path. Can be the same one that was used to run
        these experiments.
    base_experiment: name of the experiment we're comparing against.
    experimental_experiment: name of the experiment we're comparing.
    download_base_path: base path under which live the stored artifacts of
        the benchmarking experiments.
    """
    local_dir, dataset_list = download_artifacts(
        bench_config_path, base_experiment, experimental_experiment, download_base_path
    )
    metric_diffs, resource_usage_diffs = [], []
    for dataset_name in dataset_list:
        try:
            metric_diff = build_metrics_diff(dataset_name, base_experiment, experimental_experiment, local_dir)
            metric_diffs.append(metric_diff)

            base_path = os.path.join(local_dir, dataset_name, base_experiment, "resource_usage_metrics")
            experimental_path = os.path.join(local_dir, dataset_name, experimental_experiment, "resource_usage_metrics")
            resource_usage_diff = build_resource_usage_diff(
                base_path, experimental_path, base_experiment, experimental_experiment
            )
            resource_usage_diffs.append(resource_usage_diff)
        except Exception:
            logging.exception(f"Exception encountered while creating diff summary for {dataset_name}.")
    return metric_diffs, resource_usage_diffs
