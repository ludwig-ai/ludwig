import os
from contextlib import nullcontext as does_not_raise

import pytest

from ludwig.benchmarking.utils import validate_benchmarking_config
from ludwig.utils.data_utils import load_yaml


def get_benchamrking_config_validity_pair():
    local_dir = "/".join(__file__.split("/")[:-1])
    invalid_bench_configs = [
        os.path.join(local_dir, "example_files", "invalid", config_fp)
        for config_fp in os.listdir(os.path.join(local_dir, "example_files", "invalid"))
    ]
    valid_bench_configs = [
        os.path.join(local_dir, "example_files", "valid", config_fp)
        for config_fp in os.listdir(os.path.join(local_dir, "example_files", "valid"))
    ]
    return [(config_fp, does_not_raise()) for config_fp in valid_bench_configs] + [
        (config_fp, pytest.raises(ValueError)) for config_fp in invalid_bench_configs
    ]


@pytest.mark.parametrize("benchmarking_config_fp,error_expectation", get_benchamrking_config_validity_pair())
def test_benchmarking(benchmarking_config_fp, error_expectation):
    benchmarking_config = load_yaml(benchmarking_config_fp)

    with error_expectation:
        validate_benchmarking_config(benchmarking_config)
