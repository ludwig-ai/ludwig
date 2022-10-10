import os

import numpy as np
import pytest

from ludwig.benchmarking.benchmark import benchmark
from ludwig.constants import MODEL_ECD, MODEL_GBM
from ludwig.utils.data_utils import load_json

BENCHMARKING_CONFIG = """
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

    benchmark(benchmarking_config_fp)

    test_statistics_fp = os.path.join(tmpdir, dataset, experiment_name, "experiment_run", "test_statistics.json")
    test_statistics = load_json(test_statistics_fp)
    expected_test_statistics_fp = os.path.join(benchmark_directory, "expected_metrics", f"{dataset}_{model_type}.yaml")
    expected_test_statistics = load_json(expected_test_statistics_fp)

    from pprint import pprint
    print()
    pprint(test_statistics)
    print()
    pprint(expected_test_statistics)
    print()

    np.testing.assert_equal(test_statistics, expected_test_statistics)
