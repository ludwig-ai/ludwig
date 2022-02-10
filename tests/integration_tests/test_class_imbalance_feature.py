import contextlib

import pytest
import ray

import numpy as np
import pandas as pd

from ludwig.constants import OUTPUT_FLAG, PROC_COLUMN, SPLIT
from ludwig.data.preprocessing import balance_data, build_dataset, cast_columns, get_split
from ludwig.backend import initialize_backend, RAY
from ludwig.backend.base import LocalBackend
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.api import LudwigModel
from tests.integration_tests.utils import spawn

DFS = {
    "test_df_1": pd.DataFrame({"Index": np.arange(0, 200, 1),
                               "random_1": np.random.randint(0, 50, 200),
                               "random_2": np.random.choice(['Type A', 'Type B', 'Type C', 'Type D'], 200),
                               "Label": np.random.choice(2, 200, p=[0.9, 0.1])})
}

CONFIGS = {
    "test_config_1": {
        "input_features": [
            {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
            {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
            {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False}],
        "output_features": [
            {"name": "Label", "column": "Label", "type": "numerical", "output_flag": True}
        ],
        "preprocessing": {
        }
    }
}


@contextlib.contextmanager
def ray_start(num_cpus=2, num_gpus=None):
    res = ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        include_dashboard=False,
        object_store_memory=150 * 1024 * 1024,
    )
    try:
        yield res
    finally:
        ray.shutdown()


@spawn
def run_test_imbalance_ray(
        config,
        num_cpus=2,
        num_gpus=None, ):
    with ray_start(num_cpus=2, num_gpus=None):
        model = LudwigModel(config)
        train_stats, processed_df, url = model.train(DFS['test_df_1'])

    assert True is True


def run_test_imbalance_local(
        config):
    model = LudwigModel(config)
    train_stats, processed_df, url = model.train(DFS['test_df_1'])

    assert True is True  # TODO


@pytest.mark.parametrize(
    "balance", ["oversample_minority", "undersample_majority", "oversample_minority", "undersample_majority"],
)
@pytest.mark.distributed
def test_imbalance_ray(balance):
    config = CONFIGS["test_config_1"].copy()
    config['preprocessing'][balance] = 0.5
    run_test_imbalance_ray(config)


@pytest.mark.parametrize(
    "balance", ["oversample_minority", "undersample_majority", "oversample_minority", "undersample_majority"],
)
def test_imbalance_local(balance):
    config = CONFIGS["test_config_1"].copy()
    config['preprocessing'][balance] = 0.5
    run_test_imbalance_local(config)
