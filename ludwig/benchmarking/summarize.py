import argparse
import logging
import os
import shutil
from typing import List, Tuple

from ludwig.benchmarking.summary_dataclasses import (
    build_metrics_diff,
    build_resource_usage_diff,
    export_metrics_diff_to_csv,
    export_resource_usage_diff_to_csv,
    MetricsDiff,
    ResourceUsageDiff,
)
from ludwig.benchmarking.utils import download_artifacts

logger = logging.getLogger()


def summarize_metrics(
    bench_config_path: str, base_experiment: str, experimental_experiment: str, download_base_path: str
) -> Tuple[List[str], List[MetricsDiff], List[List[ResourceUsageDiff]]]:
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

            base_path = os.path.join(local_dir, dataset_name, base_experiment)
            experimental_path = os.path.join(local_dir, dataset_name, experimental_experiment)
            resource_usage_diff = build_resource_usage_diff(
                base_path, experimental_path, base_experiment, experimental_experiment
            )
            resource_usage_diffs.append(resource_usage_diff)
        except Exception:
            logger.exception(f"Exception encountered while creating diff summary for {dataset_name}.")
    shutil.rmtree(local_dir, ignore_errors=True)
    export_and_print(dataset_list, metric_diffs, resource_usage_diffs)
    return dataset_list, metric_diffs, resource_usage_diffs


def export_and_print(
    dataset_list: List[str], metric_diffs: List[MetricsDiff], resource_usage_diffs: List[List[ResourceUsageDiff]]
) -> None:
    """Export to CSV and print a diff of performance and resource usage metrics of two experiments.

    :param dataset_list: list of datasets for which to print the diffs.
    :param metric_diffs: Diffs for the performance metrics by dataset.
    :param resource_usage_diffs: Diffs for the resource usage metrics per dataset per LudwigProfiler tag.
    """
    for dataset_name, experiment_metric_diff in zip(dataset_list, metric_diffs):
        output_path = os.path.join("summarize_output", "performance_metrics", dataset_name)
        os.makedirs(output_path, exist_ok=True)

        logger.info(
            "Model performance metrics for *{}* vs. *{}* on dataset *{}*".format(
                experiment_metric_diff.base_experiment_name,
                experiment_metric_diff.experimental_experiment_name,
                experiment_metric_diff.dataset_name,
            )
        )
        logger.info(experiment_metric_diff.to_string())
        filename = (
            "-".join([experiment_metric_diff.base_experiment_name, experiment_metric_diff.experimental_experiment_name])
            + ".csv"
        )
        export_metrics_diff_to_csv(experiment_metric_diff, os.path.join(output_path, filename))

    for dataset_name, experiment_resource_diff in zip(dataset_list, resource_usage_diffs):
        output_path = os.path.join("summarize_output", "resource_usage_metrics", dataset_name)
        os.makedirs(output_path, exist_ok=True)
        for tag_diff in experiment_resource_diff:
            logger.info(
                "Resource usage for *{}* vs. *{}* on *{}* of dataset *{}*".format(
                    tag_diff.base_experiment_name,
                    tag_diff.experimental_experiment_name,
                    tag_diff.code_block_tag,
                    dataset_name,
                )
            )
            logger.info(tag_diff.to_string())
            filename = (
                "-".join(
                    [tag_diff.code_block_tag, tag_diff.base_experiment_name, tag_diff.experimental_experiment_name]
                )
                + ".csv"
            )
            export_resource_usage_diff_to_csv(tag_diff, os.path.join(output_path, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize the model performance metrics and resource usage metrics of two experiments.",
        prog="python summarize.py",
        usage="%(prog)s [options]",
    )
    parser.add_argument("--benchmarking_config", type=str, help="The benchmarking config.")
    parser.add_argument("--base_experiment", type=str, help="The name of the first experiment.")
    parser.add_argument("--experimental_experiment", type=str, help="The name of the second experiment.")
    parser.add_argument("--download_base_path", type=str, help="The base path to download experiment artifacts from.")
    args = parser.parse_args()
    summarize_metrics(
        args.benchmarking_config, args.base_experiment, args.experimental_experiment, args.download_base_path
    )
