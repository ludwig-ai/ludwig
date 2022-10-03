import os
from pprint import pprint

import pytest

from ludwig.benchmarking.benchmark import benchmark
from ludwig.utils.data_utils import load_json  # , load_yaml

BENCHMARKING_CONFIG = """
experiment_name: {experiment_name}
hyperopt: false
process_config_file_path: {process_config_file_path}
export:
  export_artifacts: true
  export_base_path: {export_base_path}
profiler:
  enable: true
  use_torch_profiler: false
  logging_interval: 0.1
experiments:
  - dataset_name: {dataset}
    config_path: {config_path}
"""

dataset_name_to_metric = {
    "ames_housing": "r2",
    "mercedes_benz_greener": "r2",
    "protein": "r2",
    "naval": "r2",
    "sarcos": "r2",
    "adult_census_income": "accuracy",
}

dataset_to_expected_performance = {
    "ames_housing": 0.75,  # 0.7565240263938904
    "mercedes_benz_greener": 0.48,  # 0.5213764905929565
    "protein": 0.47,  # 0.49943429231643677
    "sarcos": 0.94,  # 0.9894186854362488
    "naval": 0.65,  # 0.14058196544647217
    "adult_census_income": 0.81,  # 0.8540130853652954
}


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "dataset", ["ames_housing", "mercedes_benz_greener", "adult_census_income", "protein", "sarcos", "naval"]
)
def test_performance(dataset, tmpdir):
    benchmark_directory = "/".join(__file__.split("/")[:-1])
    experiment_name = "regression_test"
    config_path = os.path.join(benchmark_directory, f"{dataset}.yaml")
    process_config_file_path = os.path.join(benchmark_directory, "process_config.py")

    benchmarking_config = BENCHMARKING_CONFIG.format(
        experiment_name=experiment_name,
        process_config_file_path=process_config_file_path,
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
    # output_feature_name = load_yaml(config_path)["output_features"][0]["name"]
    # metric_name = dataset_name_to_metric[dataset]
    # expected_performance = dataset_to_expected_performance[dataset]
    pprint(test_statistics)
    # assert test_statistics[output_feature_name][metric_name] > expected_performance

    # todo (wael): enable profiler and add resource usage asserts (esp. time and memory usage)
    preprocessing_resource_usage_fp = os.path.join(
        tmpdir, dataset, experiment_name, "system_resource_usage", "preprocessing", "run_0.json"
    )
    training_resource_usage_fp = os.path.join(
        tmpdir, dataset, experiment_name, "system_resource_usage", "training", "run_0.json"
    )
    evaluation_resource_usage_fp = os.path.join(
        tmpdir, dataset, experiment_name, "system_resource_usage", "evaluation", "run_0.json"
    )

    preprocessing_resource_usage = load_json(preprocessing_resource_usage_fp)
    training_resource_usage = load_json(training_resource_usage_fp)
    evaluation_resource_usage = load_json(evaluation_resource_usage_fp)

    print()
    pprint(preprocessing_resource_usage)
    print()
    pprint(training_resource_usage)
    print()
    pprint(evaluation_resource_usage)
