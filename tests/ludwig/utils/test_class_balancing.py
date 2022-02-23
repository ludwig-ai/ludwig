import contextlib

import pytest
import ray

import numpy as np
import pandas as pd

from ludwig.constants import OUTPUT_FLAG, NAME, PROC_COLUMN
from ludwig.data.preprocessing import balance_data
from ludwig.backend import create_ray_backend
from ludwig.backend.ray import RayBackend
from ludwig.backend.base import LocalBackend
from tests.integration_tests.utils import spawn

DFS = {
    "test_df_1": pd.DataFrame({"Index": np.arange(0, 200, 1),
                               "random_1": np.random.randint(0, 50, 200),
                               "random_2": np.random.choice(['Type A', 'Type B', 'Type C', 'Type D'], 200),
                               "Label": np.concatenate((np.zeros(180), np.ones(20))),
                               "split": np.zeros(200)})
}

CONFIGS = {
    "test_config_1": {
        "input_features": [
            {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
            {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
            {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False}],
        "output_features": [
            {"name": "Label", "column": "Label", "type": "binary", "output_flag": True}
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
def run_test_balance_data_ray(
        input_df,
        input_features,
        preprocessing_parameters,
        target,
        num_cpus=2,
        num_gpus=None, ):
    with ray_start(num_cpus=num_cpus, num_gpus=num_gpus):
        backend = create_ray_backend()
        input_df = backend.df_engine.from_pandas(input_df)
        test_df = balance_data(input_df,
                               input_features,
                               preprocessing_parameters,
                               backend)

        new_class_balance = round(test_df[target].value_counts().compute()[
                                      test_df[target].value_counts().compute().idxmin()] /
                                  test_df[target].value_counts().compute()[
                                      test_df[target].value_counts().compute().idxmax()], 1)

        assert input_df.compute().shape == (200, 6)
        assert input_df[input_df['Label'] == 1].compute().shape == (20, 6)
        assert input_df[input_df['Label'] == 0].compute().shape == (180, 6)
        assert test_df.compute().shape == (270, 6)
        assert isinstance(backend, RayBackend)
        assert new_class_balance == 0.5


def run_test_balance_data_local(
        input_df,
        input_features,
        preprocessing_parameters,
        target,
        backend, ):
    test_df = balance_data(input_df,
                           input_features,
                           preprocessing_parameters,
                           backend)

    new_class_balance = round(test_df[target].value_counts()[
                                  test_df[target].value_counts().idxmin()] /
                              test_df[target].value_counts()[
                                  test_df[target].value_counts().idxmax()], 1)

    assert input_df.shape == (200, 5)
    assert input_df[input_df['Label'] == 1].shape == (20, 5)
    assert input_df[input_df['Label'] == 0].shape == (180, 5)
    assert test_df.shape == (270, 5)
    assert new_class_balance == 0.5
    assert isinstance(backend, LocalBackend)


@pytest.mark.parametrize(
    "balance", ["oversample_minority", "undersample_majority"]
)
@pytest.mark.distributed
def test_balance_data_ray(balance):
    config = CONFIGS["test_config_1"].copy()
    config['preprocessing'][balance] = 0.5
    df = DFS['test_df_1'].copy()
    features = config['input_features'] + config['output_features']
    for feature in features:
        feature[PROC_COLUMN] = feature[NAME]
    preprocessing_params = config['preprocessing']
    target = None
    for feature in features:
        if feature[OUTPUT_FLAG]:
            target = feature[NAME]

    run_test_balance_data_ray(df,
                              features,
                              preprocessing_params,
                              target)


@pytest.mark.parametrize(
    "balance", ["oversample_minority", "undersample_majority"],
)
def test_balance_data_local(balance):
    config = CONFIGS["test_config_1"].copy()
    config['preprocessing'][balance] = 0.5
    df = DFS['test_df_1'].copy()
    features = config['input_features'] + config['output_features']
    for feature in features:
        feature[PROC_COLUMN] = feature[NAME]
    preprocessing_params = config['preprocessing']
    backend = LocalBackend()
    target = None
    for feature in features:
        if feature[OUTPUT_FLAG]:
            target = feature[NAME]

    run_test_balance_data_local(df,
                                features,
                                preprocessing_params,
                                target,
                                backend)


def test_non_binary_failure():
    config = CONFIGS["test_config_1"].copy()
    config['preprocessing']["oversample_minority"] = 0.5
    config["output_features"][0]["type"] = "numerical"
    df = DFS['test_df_1'].copy()
    features = config['input_features'] + config['output_features']
    for feature in features:
        feature[PROC_COLUMN] = feature[NAME]
    preprocessing_params = config['preprocessing']
    backend = LocalBackend()
    target = None
    for feature in features:
        if feature[OUTPUT_FLAG]:
            target = feature[NAME]
    with pytest.raises(ValueError):
        run_test_balance_data_local(df,
                                    features,
                                    preprocessing_params,
                                    target,
                                    backend)


def test_multiple_class_failure():
    config = CONFIGS["test_config_1"].copy()
    config['preprocessing']["oversample_minority"] = 0.5
    config["output_features"].append({"name": "Label2", "column": "Label2", "type": "binary", "output_flag": True})
    df = DFS['test_df_1'].copy()
    df['Label2'] = np.concatenate((np.zeros(180), np.ones(20)))
    features = config['input_features'] + config['output_features']
    for feature in features:
        feature[PROC_COLUMN] = feature[NAME]
    preprocessing_params = config['preprocessing']
    backend = LocalBackend()
    target = None
    for feature in features:
        if feature[OUTPUT_FLAG]:
            target = feature[NAME]
    with pytest.raises(ValueError):
        run_test_balance_data_local(df,
                                    features,
                                    preprocessing_params,
                                    target,
                                    backend)
