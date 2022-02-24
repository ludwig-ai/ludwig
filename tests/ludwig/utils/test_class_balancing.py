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
        target_balance,
        num_cpus=2,
        num_gpus=None, ):
    with ray_start(num_cpus=num_cpus, num_gpus=num_gpus):
        backend = create_ray_backend()
        input_df = backend.df_engine.from_pandas(input_df)
        test_df = balance_data(input_df,
                               input_features,
                               preprocessing_parameters,
                               backend)

        majority_class = test_df[target].value_counts().compute()[test_df[target].value_counts().compute().idxmax()]
        minority_class = test_df[target].value_counts().compute()[test_df[target].value_counts().compute().idxmin()]
        new_class_balance = round(minority_class / majority_class, 2)

        assert (target_balance - 0.02) <= new_class_balance <= (target_balance + 0.02)
        assert isinstance(backend, RayBackend)


def run_test_balance_data_local(
        input_df,
        input_features,
        preprocessing_parameters,
        target,
        target_balance,
        backend, ):
    test_df = balance_data(input_df,
                           input_features,
                           preprocessing_parameters,
                           backend)

    majority_class = test_df[target].value_counts()[test_df[target].value_counts().idxmax()]
    minority_class = test_df[target].value_counts()[test_df[target].value_counts().idxmin()]
    new_class_balance = round(minority_class / majority_class, 2)

    assert (target_balance - 0.02) <= new_class_balance <= (target_balance + 0.02)
    assert isinstance(backend, LocalBackend)


@pytest.mark.parametrize(
    "method, balance", [
        ("oversample_minority", 0.25),
        ("oversample_minority", 0.5),
        ("oversample_minority", 0.75),
        ("undersample_majority", 0.25),
        ("undersample_majority", 0.5),
        ("undersample_majority", 0.75)
    ]
)
@pytest.mark.distributed
def test_balance_data_ray(method, balance):
    config = {"input_features": [
                {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
                {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
                {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False},
            ],
                "output_features": [{"name": "Label", "column": "Label", "type": "binary", "output_flag": True}],
                "preprocessing": {"oversample_minority": None,
                                  "undersample_majority": None},
            }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    config["preprocessing"][method] = balance
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
                              target,
                              balance)


@pytest.mark.parametrize(
    "method, balance", [
        ("oversample_minority", 0.25),
        ("oversample_minority", 0.5),
        ("oversample_minority", 0.75),
        ("undersample_majority", 0.25),
        ("undersample_majority", 0.5),
        ("undersample_majority", 0.75)
    ]
)
def test_balance_data_local(method, balance):
    config = {"input_features": [
        {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
        {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
        {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False},
    ],
        "output_features": [{"name": "Label", "column": "Label", "type": "binary", "output_flag": True}],
        "preprocessing": {"oversample_minority": None,
                          "undersample_majority": None},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    config["preprocessing"][method] = balance
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
                                balance,
                                backend)


def test_non_binary_failure():
    config = {"input_features": [
        {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
        {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
        {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False},
    ],
        "output_features": [{"name": "Label", "column": "Label", "type": "number", "output_flag": True}],
        "preprocessing": {},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    config['preprocessing']["oversample_minority"] = 0.5
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
                                    0.5,
                                    backend)


def test_multiple_class_failure():
    config = {"input_features": [
        {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
        {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
        {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False},
    ],
        "output_features": [{"name": "Label", "column": "Label", "type": "binary", "output_flag": True},
                            {"name": "Label2", "column": "Label2", "type": "binary", "output_flag": True}],
        "preprocessing": {},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "Label2": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    config['preprocessing']["oversample_minority"] = 0.5
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
                                    0.5,
                                    backend)
