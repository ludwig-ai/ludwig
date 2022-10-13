import os
from contextlib import nullcontext as does_not_raise

import pytest

from ludwig.benchmarking.utils import validate_benchmarking_config
from ludwig.utils.data_utils import load_yaml


def get_benchamrking_configs(validity):
    local_dir = "/".join(__file__.split("/")[:-1])
    return [
        os.path.join(local_dir, "example_files", validity, config_fp)
        for config_fp in os.listdir(os.path.join(local_dir, "example_files", validity))
    ]


@pytest.mark.parametrize("benchmarking_config_fp", get_benchamrking_configs("valid"))
def test_valid_benchmarking_configs_valid(benchmarking_config_fp):
    benchmarking_config = load_yaml(benchmarking_config_fp)

    with does_not_raise():
        validate_benchmarking_config(benchmarking_config)


@pytest.mark.parametrize("benchmarking_config_fp", get_benchamrking_configs("invalid"))
def test_invalid_benchmarking_configs_valid(benchmarking_config_fp):
    benchmarking_config = load_yaml(benchmarking_config_fp)

    with pytest.raises(ValueError):
        validate_benchmarking_config(benchmarking_config)
