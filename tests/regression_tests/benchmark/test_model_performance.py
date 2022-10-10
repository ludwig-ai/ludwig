import os

import pytest

from ludwig.benchmarking.benchmark import benchmark
from ludwig.constants import MODEL_ECD, MODEL_GBM
from ludwig.utils.data_utils import load_json, load_yaml

BENCHMARKING_CONFIG = """
experiment_name: {experiment_name}
hyperopt: false
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
    "ames_housing": 0.75,
    "mercedes_benz_greener": 0.48,
    "adult_census_income": 0.81,
    "protein": 0.47,
    "sarcos": 0.94,
    "naval": 0.65,
}

dataset_to_expected_preprocessing_time = {  # in microseconds
    "ames_housing": 6e5,
    "mercedes_benz_greener": 3.5e6,
    "adult_census_income": 2e6,
    "protein": 7e5,
    "sarcos": 2.4e6,
    "naval": 6e5,
}

dataset_to_expected_training_time = {  # in microseconds
    "ames_housing": 3e7,
    "mercedes_benz_greener": 1.3e8,
    "adult_census_income": 6e7,
    "protein": 1.25e8,
    "sarcos": 3e8,
    "naval": 8e8,
}

dataset_to_expected_evaluation_time = {  # in microseconds
    "ames_housing": 1.3e5,
    "mercedes_benz_greener": 4.5e5,
    "adult_census_income": 6e5,
    "protein": 2.5e5,
    "sarcos": 3.5e5,
    "naval": 1.3e5,
}


@pytest.mark.benchmark
@pytest.mark.parametrize("model_type", [MODEL_GBM, MODEL_ECD])
@pytest.mark.parametrize(
    "dataset", ["ames_housing", "mercedes_benz_greener", "adult_census_income", "protein", "sarcos", "naval"]
)
def test_performance(model_type, dataset, tmpdir):
    benchmark_directory = "/".join(__file__.split("/")[:-1])
    experiment_name = "regression_test"
    config_path = os.path.join(benchmark_directory, "configs", f"{dataset}_{model_type}.yaml")

    benchmarking_config = BENCHMARKING_CONFIG.format(
        experiment_name=experiment_name,
        export_base_path=tmpdir,
        dataset=dataset,
        config_path=config_path,
    )

    benchmarking_config_fp = tmpdir + "/benchmarking_config.yaml"
    with open(benchmarking_config_fp, "w") as f:
        f.write(benchmarking_config)

    for _ in range(3):
        benchmark(benchmarking_config_fp)

        test_statistics_fp = os.path.join(tmpdir, dataset, experiment_name, "experiment_run", "test_statistics.json")
        test_statistics = load_json(test_statistics_fp)
        output_feature_name = load_yaml(config_path)["output_features"][0]["name"]
        metric_name = dataset_name_to_metric[dataset]

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
        print(dataset, model_type)
        print(metric_name, test_statistics[output_feature_name][metric_name])
        print("preprocessing_time", preprocessing_resource_usage["total_execution_time"])
        print("training_time", training_resource_usage["total_execution_time"])
        print("evaluation_time", evaluation_resource_usage["total_execution_time"])
        print()

        import shutil

        shutil.rmtree(os.path.join(tmpdir, dataset), ignore_errors=True)

    # assert test_statistics[output_feature_name][metric_name] > dataset_to_expected_performance[dataset]
    # assert preprocessing_resource_usage["total_execution_time"] < dataset_to_expected_preprocessing_time[dataset]
    # assert training_resource_usage["total_execution_time"] < dataset_to_expected_training_time[dataset]
    # assert evaluation_resource_usage["total_execution_time"] < dataset_to_expected_evaluation_time[dataset]
