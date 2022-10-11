import os
import pytest

from typing import List
from expected_metric import ExpectedMetric, MetricRegressionDirection
from ludwig.benchmarking.benchmark import benchmark
from ludwig.utils.data_utils import load_yaml


def get_test_config_paths() -> List[str]:
    """Return list of the config filenames used for benchmarking."""
    benchmark_directory = "/".join(__file__.split("/")[:-1])
    return [config_fp for config_fp in os.listdir(benchmark_directory)]


def get_dataset_from_config_path(config_path: str) -> str:
    """path/to/config/<dataset>.<descriptors>.yaml -> dataset"""
    return os.path.basename(config_path).split(".")[0]


@pytest.mark.benchmark
@pytest.mark.parametrize("config_filename", get_test_config_paths())
def test_performance(config_filename, tmpdir):
    benchmark_directory = "/".join(__file__.split("/")[:-1])
    config_path = os.path.join(benchmark_directory, "configs", config_filename)
    expected_test_statistics_fp = os.path.join(benchmark_directory, "expected_metrics", config_filename)

    benchmarking_config = {
        "experiment_name": "regression_test",
        "export": {
            "export_artifacts": True,
            "export_base_path": tmpdir
        },
        "experiments": [
            {
                "dataset_name": get_dataset_from_config_path(config_path),
                "config_path": config_path
            }
        ]
    }

    benchmarking_artifacts = benchmark(benchmarking_config)

    for artifact in benchmarking_artifacts:
        expected_metrics_dict = load_yaml(expected_test_statistics_fp)
        expected_metrics: List[ExpectedMetric] = [
            ExpectedMetric.from_dict(expected_metric) for expected_metric in expected_metrics_dict["metrics"]
        ]
        for expected_metric in expected_metrics:
            tolerance = (
                expected_metric.percent_change_sensitivity
                * expected_metric.expected_value
                * expected_metric.regression_direction
            )
            if expected_metric.regression_direction == MetricRegressionDirection.LOWER.value:
                assert (
                    artifact.test_statistics[expected_metric.output_feature_name][expected_metric.metric_name]
                    >= expected_metric.expected_value + tolerance
                )
            else:
                assert (
                    artifact.test_statistics[expected_metric.output_feature_name][expected_metric.metric_name]
                    <= expected_metric.expected_value + tolerance
                )
