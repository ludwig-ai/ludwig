import os
import pytest

from typing import List
from expected_metric import ExpectedMetric, MetricRegressionDirection
from ludwig.benchmarking.benchmark import benchmark
from ludwig.constants import MODEL_ECD, MODEL_GBM
from ludwig.utils.data_utils import load_json, load_yaml

BENCHMARKING_CONFIG_TEMPLATE = """
experiment_name: {experiment_name}
hyperopt: false
export:
  export_artifacts: true
  export_base_path: {export_base_path}
profiler:
  enable: false
  use_torch_profiler: false
  logging_interval: 0.1
experiments:
  - dataset_name: {dataset}
    config_path: {config_path}
"""


@pytest.mark.benchmark
@pytest.mark.parametrize("model_type", [MODEL_GBM, MODEL_ECD])
@pytest.mark.parametrize("dataset", ["ames_housing", "mercedes_benz_greener", "adult_census_income", "sarcos"])
def test_performance(model_type, dataset, tmpdir):
    benchmark_directory = "/".join(__file__.split("/")[:-1])
    experiment_name = "regression_test"
    config_path = os.path.join(benchmark_directory, "configs", f"{dataset}_{model_type}.yaml")

    benchmarking_config = BENCHMARKING_CONFIG_TEMPLATE.format(
        experiment_name=experiment_name,
        export_base_path=tmpdir,
        dataset=dataset,
        config_path=config_path,
    )

    benchmarking_config_fp = tmpdir + "/benchmarking_config.yaml"
    with open(benchmarking_config_fp, "w") as f:
        f.write(benchmarking_config)

    benchmark(benchmarking_config_fp)

    test_statistics_fp = os.path.join(tmpdir, dataset, experiment_name, "experiment_run", "test_statistics.json")
    test_statistics = load_json(test_statistics_fp)

    expected_test_statistics_fp = os.path.join(benchmark_directory, "expected_metrics", f"{dataset}_{model_type}.yaml")
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
                test_statistics[expected_metric.output_feature_name][expected_metric.metric_name]
                >= expected_metric.expected_value + tolerance
            )
        else:
            assert (
                test_statistics[expected_metric.output_feature_name][expected_metric.metric_name]
                <= expected_metric.expected_value + tolerance
            )
