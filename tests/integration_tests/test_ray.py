# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import contextlib
import os
import tempfile
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest
import ray
import torch

from ludwig.api import LudwigModel
from ludwig.backend import create_ray_backend, LOCAL_BACKEND
from ludwig.backend.ray import get_trainer_kwargs, RayBackend
from ludwig.constants import BACKFILL, BALANCE_PERCENTAGE_TOLERANCE, NAME, TRAINER
from ludwig.data.dataframe.dask import DaskEngine
from ludwig.data.preprocessing import balance_data
from ludwig.utils.data_utils import read_parquet
from tests.integration_tests.utils import (
    audio_feature,
    bag_feature,
    binary_feature,
    category_feature,
    create_data_set_to_use,
    date_feature,
    generate_data,
    h3_feature,
    image_feature,
    number_feature,
    sequence_feature,
    set_feature,
    spawn,
    text_feature,
    timeseries_feature,
    train_with_backend,
    vector_feature,
)

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


def run_api_experiment(config, dataset, backend_config, skip_save_processed_input=True):
    # Sanity check that we get 4 slots over 1 host
    kwargs = get_trainer_kwargs()
    if torch.cuda.device_count() > 0:
        assert kwargs.get("num_workers") == torch.cuda.device_count(), kwargs
        assert kwargs.get("use_gpu"), kwargs
    else:
        assert kwargs.get("num_workers") == 1, kwargs
        assert not kwargs.get("use_gpu"), kwargs

    # Train on Parquet
    model = train_with_backend(
        backend_config,
        config,
        dataset=dataset,
        evaluate=True,
        predict=False,
        skip_save_processed_input=skip_save_processed_input,
    )

    assert isinstance(model.backend, RayBackend)
    if isinstance(model.backend.df_engine, DaskEngine):
        assert model.backend.df_engine.parallelism == backend_config["processor"]["parallelism"]

    return model


def run_split_api_experiment(config, data_parquet, backend_config):
    train_fname, val_fname, test_fname = split(data_parquet)

    # Train
    train_with_backend(backend_config, config, training_set=train_fname, evaluate=False, predict=True)

    # Train + Validation
    train_with_backend(
        backend_config, config, training_set=train_fname, validation_set=val_fname, evaluate=False, predict=False
    )

    # Train + Validation + Test
    train_with_backend(
        backend_config,
        config,
        training_set=train_fname,
        validation_set=val_fname,
        test_set=test_fname,
        evaluate=False,
        predict=False,
    )


def split(data_parquet):
    data_df = read_parquet(data_parquet, LOCAL_BACKEND.df_engine.df_lib)
    train_df = data_df.sample(frac=0.8)
    test_df = data_df.drop(train_df.index).sample(frac=0.5)
    validation_df = data_df.drop(train_df.index).drop(test_df.index)

    basename, ext = os.path.splitext(data_parquet)
    train_fname = basename + ".train" + ext
    val_fname = basename + ".validation" + ext
    test_fname = basename + ".test" + ext

    train_df.to_parquet(train_fname)
    validation_df.to_parquet(val_fname)
    test_df.to_parquet(test_fname)
    return train_fname, val_fname, test_fname


@spawn
def run_test_with_features(
    input_features,
    output_features,
    num_examples=100,
    run_fn=run_api_experiment,
    expect_error=False,
    num_cpus=2,
    num_gpus=None,
    df_engine=None,
    dataset_type="parquet",
    skip_save_processed_input=True,
    nan_percent=0.0,
):
    with ray_start(num_cpus=num_cpus, num_gpus=num_gpus):
        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "output_size": 14},
            TRAINER: {"epochs": 2, "batch_size": 8},
        }

        backend_config = {**RAY_BACKEND_CONFIG}
        if df_engine:
            backend_config["processor"]["type"] = df_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_filename = os.path.join(tmpdir, "dataset.csv")
            dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=num_examples)
            dataset = create_data_set_to_use(dataset_type, dataset_csv, nan_percent=nan_percent)

            if expect_error:
                with pytest.raises(ValueError):
                    run_fn(
                        config,
                        dataset=dataset,
                        backend_config=backend_config,
                        skip_save_processed_input=skip_save_processed_input,
                    )
            else:
                run_fn(
                    config,
                    dataset=dataset,
                    backend_config=backend_config,
                    skip_save_processed_input=skip_save_processed_input,
                )


@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
@pytest.mark.distributed
def test_ray_save_processed_input(dataset_type):
    input_features = [
        category_feature(vocab_size=2, reduce_input="sum"),
    ]
    output_features = [
        category_feature(vocab_size=5),  # Regression test for #1991 requires multi-class predictions.
    ]
    run_test_with_features(
        input_features,
        output_features,
        df_engine="dask",
        dataset_type=dataset_type,
        skip_save_processed_input=False,
        nan_percent=0.1,
    )


@pytest.mark.parametrize("df_engine", ["dask", "modin"])
@pytest.mark.distributed
def test_ray_tabular(df_engine):
    input_features = [
        sequence_feature(reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum"),
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
        bag_feature(),
        vector_feature(),
        h3_feature(),
        date_feature(),
    ]
    output_features = [
        binary_feature(bool2str=["No", "Yes"]),
        binary_feature(),
        number_feature(normalization="zscore"),
    ]
    run_test_with_features(
        input_features,
        output_features,
        df_engine=df_engine,
    )


@pytest.mark.skip(reason="TODO torch")
@pytest.mark.distributed
def test_ray_text():
    input_features = [
        text_feature(),
    ]
    output_features = [
        text_feature(reduce_input=None, decoder="tagger"),
    ]
    run_test_with_features(input_features, output_features)


@pytest.mark.skip(reason="TODO torch")
@pytest.mark.distributed
def test_ray_sequence():
    input_features = [sequence_feature(max_len=10, encoder="rnn", cell_type="lstm", reduce_output=None)]
    output_features = [sequence_feature(max_len=10, decoder="tagger", attention=False, reduce_input=None)]
    run_test_with_features(input_features, output_features)


@pytest.mark.parametrize("feature_type", ["raw", "stft", "stft_phase", "group_delay", "fbank"])
@pytest.mark.distributed
def test_ray_audio(feature_type):
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessing_params = {
            "audio_file_length_limit_in_s": 3.0,
            "missing_value_strategy": BACKFILL,
            "in_memory": True,
            "padding_value": 0,
            "norm": "per_file",
            "audio_feature": {
                "type": feature_type,
                "window_length_in_s": 0.04,
                "window_shift_in_s": 0.02,
                "num_filter_bands": 80,
            },
        }
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features = [audio_feature(folder=audio_dest_folder, preprocessing=preprocessing_params)]
        output_features = [binary_feature()]
        run_test_with_features(input_features, output_features)


@pytest.mark.distributed
def test_ray_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, "generated_images")
        input_features = [
            image_feature(
                folder=image_dest_folder,
                encoder="resnet",
                preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
                output_size=16,
                num_filters=8,
            ),
        ]
        output_features = [binary_feature()]
        run_test_with_features(input_features, output_features)


@pytest.mark.skip(reason="flaky: ray is running out of resources")
@pytest.mark.distributed
def test_ray_split():
    input_features = [
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
    ]
    output_features = [category_feature(vocab_size=2, reduce_input="sum")]
    run_test_with_features(
        input_features,
        output_features,
        run_fn=run_split_api_experiment,
        num_cpus=4,
    )


@pytest.mark.distributed
def test_ray_timeseries():
    input_features = [timeseries_feature()]
    output_features = [number_feature()]
    run_test_with_features(input_features, output_features)


@pytest.mark.distributed
def test_ray_lazy_load_audio_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features = [
            audio_feature(
                folder=audio_dest_folder,
                preprocessing={
                    "in_memory": False,
                },
            )
        ]
        output_features = [binary_feature()]
        run_test_with_features(input_features, output_features, expect_error=True)


@pytest.mark.distributed
def test_ray_lazy_load_image_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, "generated_images")
        input_features = [
            image_feature(
                folder=image_dest_folder,
                encoder="resnet",
                preprocessing={"in_memory": False, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
                output_size=16,
                num_filters=8,
            ),
        ]
        output_features = [binary_feature()]
        run_test_with_features(input_features, output_features, expect_error=True)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
@pytest.mark.distributed
def test_train_gpu_load_cpu():
    input_features = [
        category_feature(vocab_size=2, reduce_input="sum"),
        number_feature(normalization="zscore"),
    ]
    output_features = [
        binary_feature(),
    ]
    run_test_with_features(input_features, output_features, run_fn=_run_train_gpu_load_cpu, num_gpus=1)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "method, balance",
    [
        ("oversample_minority", 0.25),
        ("oversample_minority", 0.5),
        ("oversample_minority", 0.75),
        ("undersample_majority", 0.25),
        ("undersample_majority", 0.5),
        ("undersample_majority", 0.75),
    ],
)
def test_balance_ray(method, balance):
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "number"},
            {"name": "random_1", "proc_column": "random_1", "type": "number"},
            {"name": "random_2", "proc_column": "random_2", "type": "number"},
        ],
        "output_features": [{"name": "Label", "proc_column": "Label", "type": "binary"}],
        "preprocessing": {"oversample_minority": None, "undersample_majority": None},
    }
    input_df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )
    config["preprocessing"][method] = balance
    target = config["output_features"][0][NAME]

    with ray_start(num_cpus=2, num_gpus=None):
        backend = create_ray_backend()
        input_df = backend.df_engine.from_pandas(input_df)
        test_df = balance_data(input_df, config["output_features"], config["preprocessing"], backend)

        majority_class = test_df[target].value_counts().compute()[test_df[target].value_counts().compute().idxmax()]
        minority_class = test_df[target].value_counts().compute()[test_df[target].value_counts().compute().idxmin()]
        new_class_balance = round(minority_class / majority_class, 2)

        assert abs(balance - new_class_balance) < BALANCE_PERCENTAGE_TOLERANCE


def _run_train_gpu_load_cpu(config, data_parquet):
    with tempfile.TemporaryDirectory() as output_dir:
        model_dir = ray.get(train_gpu.remote(config, data_parquet, output_dir))
        ray.get(predict_cpu.remote(model_dir, data_parquet))


@ray.remote(num_cpus=1, num_gpus=1)
def train_gpu(config, dataset, output_directory):
    model = LudwigModel(config, backend="local")
    _, _, output_dir = model.train(dataset, output_directory=output_directory)
    return os.path.join(output_dir, "model")


@ray.remote(num_cpus=1, num_gpus=0)
def predict_cpu(model_dir, dataset):
    model = LudwigModel.load(model_dir, backend="local")
    model.predict(dataset)


@pytest.mark.skipif(LooseVersion(ray.__version__) < LooseVersion("1.12"), reason="Serialization issue before Ray 1.12")
@pytest.mark.distributed
def test_tune_batch_size_lr():
    with ray_start(num_cpus=2, num_gpus=None):
        config = {
            "input_features": [
                number_feature(normalization="zscore"),
                set_feature(),
                binary_feature(),
            ],
            "output_features": [category_feature(vocab_size=2, reduce_input="sum")],
            "combiner": {"type": "concat", "output_size": 14},
            TRAINER: {"epochs": 2, "batch_size": "auto", "learning_rate": "auto"},
        }

        backend_config = {**RAY_BACKEND_CONFIG}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_filename = os.path.join(tmpdir, "dataset.csv")
            dataset_csv = generate_data(
                config["input_features"], config["output_features"], csv_filename, num_examples=100
            )
            dataset_parquet = create_data_set_to_use("parquet", dataset_csv)
            model = run_api_experiment(config, dataset=dataset_parquet, backend_config=backend_config)
            assert model.config[TRAINER]["batch_size"] != "auto"
            assert model.config[TRAINER]["learning_rate"] != "auto"
