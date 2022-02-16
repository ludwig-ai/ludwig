import contextlib

import pytest
import ray
import tempfile
import os

import numpy as np
import pandas as pd

from ludwig.constants import OUTPUT_FLAG, PROC_COLUMN, SPLIT
from ludwig.data.preprocessing import balance_data, build_dataset, cast_columns, get_split
from ludwig.backend import create_ray_backend
from ludwig.backend.base import LocalBackend
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.api import LudwigModel
from tests.integration_tests.utils import spawn
from tests.integration_tests.utils import create_data_set_to_use
from ray.data import from_dask
import dask.dataframe as dd

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
        num_gpus=None, ):
    with ray_start(num_cpus=num_cpus, num_gpus=num_gpus):
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     csv_filename = os.path.join(tmpdir, "dataset.csv")
        #     input_df.to_csv(csv_filename)
        #     dataset_parquet = create_data_set_to_use("parquet", csv_filename)

        backend = create_ray_backend()
        model = LudwigModel(config, backend=backend)
        # _, processed_data_file, _ = model.train(dataset=dataset_parquet,
        #                                         skip_save_model=True,
        #                                         skip_save_log=True,
        #                                         skip_save_progress=True,
        #                                         skip_save_processed_input=True,
        #                                         skip_save_training_description=True,
        #                                         skip_save_training_statistics=True)

        _, processed_data_df, _ = model.train(input_df,
                                              skip_save_model=True,
                                              skip_save_log=True,
                                              skip_save_progress=True,
                                              skip_save_processed_input=True,
                                              skip_save_training_description=True,
                                              skip_save_training_statistics=True)

        if balance == 'oversample_minority':
            input_train_set = input_df.sample(frac=0.7, replace=False)
            processed_train_set_df = processed_data_df[0].dataset
            # processed_train_set_file = processed_data_file[0].dataset
            assert len(input_train_set) < processed_data_df[0].size
            # assert len(input_train_set) < processed_data_file[0].size
            assert len(input_train_set) == 140
            assert 0.05 <= (len(input_train_set[input_train_set['Label'] == 1]) / len(input_train_set)) <= 0.15
            assert round(sum(processed_train_set_df['Label_mZFLky']) /
                         (len(processed_train_set_df['Label_mZFLky']) - sum(processed_train_set_df['Label_mZFLky'])),
                         1) == 0.5
            # assert round(sum(processed_train_set_file['Label_mZFLky']) /
            #              (len(processed_train_set_file['Label_mZFLky']) - sum(processed_train_set_file['Label_mZFLky'])),
            #              1) == 0.5
            assert 60 <= sum(processed_train_set_df['Label_mZFLky']) <= 70
            # assert 60 <= sum(processed_train_set_file['Label_mZFLky']) <= 70
            assert 120 <= (len(processed_train_set_df['Label_mZFLky']) - sum(processed_train_set_df['Label_mZFLky'])) <= 140
            # assert 120 <= (len(processed_train_set_file['Label_mZFLky']) - sum(processed_train_set_file['Label_mZFLky'])) <= 140

        if balance == 'undersample_majority':
            input_train_set = input_df.sample(frac=0.7, replace=False)
            processed_train_set_df = processed_data_df[0].dataset
            # processed_train_set_file = processed_data_file[0].dataset
            assert len(input_train_set) < processed_data_df[0].size
            # assert len(input_train_set) < processed_data_file[0].size
            assert len(input_train_set) == 140
            assert 0.05 <= len(input_train_set[input_train_set['Label'] == 1]) / len(input_train_set) <= 0.15
            assert round(sum(processed_train_set_df['Label_mZFLky']) /
                         (len(processed_train_set_df['Label_mZFLky']) - sum(processed_train_set_df['Label_mZFLky'])),
                         1) == 0.5
            # assert round(sum(processed_train_set_file['Label_mZFLky']) /
            #              (len(processed_train_set_file['Label_mZFLky']) - sum(processed_train_set_file['Label_mZFLky'])),
            #              1) == 0.5
            assert 7 <= sum(processed_train_set_df['Label_mZFLky']) <= 17
            # assert 7 <= sum(processed_train_set_file['Label_mZFLky']) <= 17
            assert 14 <= (len(processed_train_set_df['Label_mZFLky']) - sum(processed_train_set_df['Label_mZFLky'])) <= 34
            # assert 14 <= (len(processed_train_set_file['Label_mZFLky']) - sum(processed_train_set_file['Label_mZFLky'])) <= 34


def run_test_imbalance_local(
        input_df,
        config,
        balance, ):
    model = LudwigModel(config)
    train_stats, processed_df, url = model.train(input_df,
                                                 skip_save_model=True,
                                                 skip_save_log=True,
                                                 skip_save_progress=True,
                                                 skip_save_processed_input=True,
                                                 skip_save_training_description=True,
                                                 skip_save_training_statistics=True)

    if balance == 'oversample_minority':
        input_train_set = input_df.sample(frac=0.7, replace=False)
        processed_train_set = processed_df[0].dataset
        assert len(input_train_set) < processed_df[0].size
        assert len(input_train_set) == 140
        assert 0.05 <= (len(input_train_set[input_train_set['Label'] == 1]) / len(input_train_set)) <= 0.15
        assert round(sum(processed_train_set['Label_mZFLky']) /
                     (len(processed_train_set['Label_mZFLky']) - sum(processed_train_set['Label_mZFLky'])), 1) == 0.5
        assert 60 <= sum(processed_train_set['Label_mZFLky']) <= 70
        assert 120 <= (len(processed_train_set['Label_mZFLky']) - sum(processed_train_set['Label_mZFLky'])) <= 140

    if balance == 'undersample_majority':
        input_train_set = input_df.sample(frac=0.7, replace=False)
        processed_train_set = processed_df[0].dataset
        assert len(input_train_set) > processed_df[0].size
        assert len(input_train_set) == 140
        assert 0.05 <= len(input_train_set[input_train_set['Label'] == 1]) / len(input_train_set) <= 0.15
        assert round(sum(processed_train_set['Label_mZFLky']) /
                     (len(processed_train_set['Label_mZFLky']) - sum(processed_train_set['Label_mZFLky'])), 1) == 0.5
        assert 7 <= sum(processed_train_set['Label_mZFLky']) <= 17
        assert 14 <= (len(processed_train_set['Label_mZFLky']) - sum(processed_train_set['Label_mZFLky'])) <= 34


@pytest.mark.parametrize(
    "balance", ["oversample_minority", "undersample_majority"],
)
@pytest.mark.distributed
def test_imbalance_ray(balance):
    df = DFS['test_df_1']
    config = CONFIGS["test_config_1"].copy()
    if balance == "oversample_minority":
        config['preprocessing'][balance] = 0.5
        config['preprocessing']["undersample_majority"] = None
    else:
        config['preprocessing'][balance] = 0.5
        config['preprocessing']["oversample_minority"] = None
    run_test_imbalance_ray(df, config, balance)


@pytest.mark.parametrize(
    "balance", ["oversample_minority", "undersample_majority"],
)
def test_imbalance_local(balance):
    df = DFS['test_df_1']
    config = CONFIGS["test_config_1"].copy()
    if balance == "oversample_minority":
        config['preprocessing'][balance] = 0.5
        config['preprocessing']["undersample_majority"] = None
    else:
        config['preprocessing'][balance] = 0.5
        config['preprocessing']["oversample_minority"] = None
    run_test_imbalance_local(df, config, balance)
