import contextlib
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import ray

from ludwig.api import LudwigModel
from ludwig.backend import LocalBackend
from ludwig.backend.ray import RayBackend
from tests.integration_tests.utils import create_data_set_to_use, spawn

rs = np.random.RandomState(42)
RAY_BACKEND_CONFIG = {
    "type": "ray",
    "processor": {
        "parallelism": 2,
    },
    "trainer": {
        "use_gpu": False,
        "num_workers": 2,
        "resources_per_worker": {
            "CPU": 0.1,
            "GPU": 0,
        },
    },
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
    input_df,
    config,
    balance,
    num_cpus=2,
    num_gpus=None,
):
    with ray_start(num_cpus=num_cpus, num_gpus=num_gpus):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_filename = os.path.join(tmpdir, "dataset.csv")
            input_df.to_csv(csv_filename)
            dataset_parquet = create_data_set_to_use("parquet", csv_filename)

            model = LudwigModel(config, backend=RAY_BACKEND_CONFIG, callbacks=None)
            output_dir = None

            try:
                _, output_dataset, output_dir = model.train(
                    dataset=dataset_parquet,
                    training_set=None,
                    validation_set=None,
                    test_set=None,
                    skip_save_processed_input=True,
                    skip_save_progress=True,
                    skip_save_unprocessed_output=True,
                    skip_save_log=True,
                )
            finally:
                # Remove results/intermediate data saved to disk
                shutil.rmtree(output_dir, ignore_errors=True)

            input_train_set = input_df.sample(frac=0.7, replace=False)
            processed_len = output_dataset[0].ds.count()
            processed_target_pos = output_dataset[0].ds.sum(on="Label_mZFLky")
            processed_target_neg = output_dataset[0].ds.count() - output_dataset[0].ds.sum(on="Label_mZFLky")
            assert len(input_train_set) == 140
            assert 0.05 <= len(input_train_set[input_train_set["Label"] == 1]) / len(input_train_set) <= 0.15
            assert round(processed_target_pos / processed_target_neg, 1) == 0.5
            assert model.backend.df_engine.parallelism == RAY_BACKEND_CONFIG["processor"]["parallelism"]
            assert isinstance(model.backend, RayBackend)

        if balance == "oversample_minority":
            assert len(input_train_set) < processed_len

        if balance == "undersample_majority":
            assert len(input_train_set) > processed_len


def run_test_imbalance_local(
    input_df,
    config,
    balance,
):
    model = LudwigModel(config)
    _, output_dataset, output_dir = model.train(
        input_df,
        skip_save_model=True,
        skip_save_log=True,
        skip_save_progress=True,
        skip_save_processed_input=True,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
    )

    input_train_set = input_df.sample(frac=0.7, replace=False)
    processed_len = output_dataset[0].size
    processed_target_pos = sum(output_dataset[0].dataset["Label_mZFLky"])
    processed_target_neg = len(output_dataset[0].dataset["Label_mZFLky"]) - processed_target_pos
    assert len(input_train_set) == 140
    assert 0.05 <= len(input_train_set[input_train_set["Label"] == 1]) / len(input_train_set) <= 0.15
    assert round(processed_target_pos / processed_target_neg, 1) == 0.5
    assert isinstance(model.backend, LocalBackend)

    if balance == "oversample_minority":
        assert len(input_train_set) < processed_len
        assert 55 <= processed_target_pos <= 75
        assert 110 <= processed_target_neg <= 150

    if balance == "undersample_majority":
        assert len(input_train_set) > processed_len
        assert 7 <= processed_target_pos <= 20
        assert 14 <= processed_target_neg <= 40


@pytest.mark.parametrize(
    "balance",
    ["oversample_minority", "undersample_majority"],
)
@pytest.mark.distributed
@pytest.mark.skip(reason="Flaky")
def test_imbalance_ray(balance):
    config = {
        "input_features": [
            {"name": "Index", "column": "Index", "type": "numerical"},
            {"name": "random_1", "column": "random_1", "type": "numerical"},
            {"name": "random_2", "column": "random_2", "type": "numerical"},
        ],
        "output_features": [{"name": "Label", "column": "Label", "type": "binary"}],
        "trainer": {"epochs": 2, "batch_size": 8},
        "preprocessing": {},
    }
    split_col = np.concatenate((np.zeros(140), np.ones(20), np.full(40, 2)))
    rs.shuffle(split_col)
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": split_col,
        }
    )

    config["preprocessing"][balance] = 0.5
    run_test_imbalance_ray(df, config, balance)


@pytest.mark.parametrize(
    "balance",
    ["oversample_minority", "undersample_majority"],
)
def test_imbalance_local(balance):
    config = {
        "input_features": [
            {"name": "Index", "column": "Index", "type": "numerical"},
            {"name": "random_1", "column": "random_1", "type": "numerical"},
            {"name": "random_2", "column": "random_2", "type": "numerical"},
        ],
        "output_features": [{"name": "Label", "column": "Label", "type": "binary"}],
        "trainer": {"epochs": 2, "batch_size": 8},
        "preprocessing": {},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
        }
    )

    config["preprocessing"][balance] = 0.5
    run_test_imbalance_local(df, config, balance)
