import os
from typing import List

import pytest
from expected_metric import ExpectedMetric

from ludwig.benchmarking.benchmark import benchmark
from ludwig.utils.data_utils import load_yaml

SKIPPED_CONFIG_ISSUES = {
    "mercedes_benz_greener.ecd.yaml": "https://github.com/ludwig-ai/ludwig/issues/2978",
    "sarcos.ecd.yaml": "https://github.com/ludwig-ai/ludwig/issues/3019",
    "sarcos.gbm.yaml": "https://github.com/ludwig-ai/ludwig/issues/3019",
}


def get_test_config_filenames() -> List[str]:
    """Return list of the config filenames used for benchmarking."""
    benchmark_directory = "/".join(__file__.split("/")[:-1] + ["configs"])
    return [config_fp for config_fp in os.listdir(benchmark_directory)]


def get_dataset_from_config_path(config_path: str) -> str:
    """path/to/config/<dataset>.<descriptors>.yaml -> dataset."""
    return os.path.basename(config_path).split(".")[0]


@pytest.mark.benchmark
@pytest.mark.parametrize("config_filename", get_test_config_filenames())
def test_performance(config_filename, tmpdir):
    if config_filename in SKIPPED_CONFIG_ISSUES:
        pytest.skip(reason=SKIPPED_CONFIG_ISSUES[config_filename])
        return

    benchmark_directory = "/".join(__file__.split("/")[:-1])
    config_path = os.path.join(benchmark_directory, "configs", config_filename)
    expected_test_statistics_fp = os.path.join(benchmark_directory, "expected_metrics", config_filename)
    dataset_name = get_dataset_from_config_path(config_path)

    if not os.path.exists(expected_test_statistics_fp):
        raise FileNotFoundError(
            """No corresponding expected metrics found for benchmarking config '{config_path}'.
            Please add a new metrics YAML file '{expected_test_statistics_fp}'. Suggested content:

            metrics:
              - output_feature_name: <YOUR_OUTPUT_FEATURE e.g. SalePrice>
                metric_name: <YOUR METRIC NAME e.g. accuracy>
                expected_value: <A FLOAT VALUE>
                tolerance_percent: 0.15"""
        )
    expected_metrics_dict = load_yaml(expected_test_statistics_fp)

    benchmarking_config = {
        "experiment_name": "regression_test",
        "export": {"export_artifacts": True, "export_base_path": tmpdir},
        "experiments": [{"dataset_name": dataset_name, "config_path": config_path}],
    }
    benchmarking_artifacts = benchmark(benchmarking_config)
    experiment_artifact, err = benchmarking_artifacts[dataset_name]
    if err is not None:
        raise err

    expected_metrics: List[ExpectedMetric] = [
        ExpectedMetric.from_dict(expected_metric) for expected_metric in expected_metrics_dict["metrics"]
    ]
    for expected_metric in expected_metrics:
        tolerance = expected_metric.tolerance_percentage * expected_metric.expected_value
        output_feature_name = expected_metric.output_feature_name
        metric_name = expected_metric.metric_name
        experiment_metric_value = experiment_artifact.test_statistics[output_feature_name][metric_name]
        assert abs(expected_metric.expected_value - experiment_metric_value) <= tolerance, (
            f"The obtained {metric_name} value ({experiment_metric_value}) was not within"
            f" {100 * expected_metric.tolerance_percentage}% of the expected value ({expected_metric.expected_value})."
        )
