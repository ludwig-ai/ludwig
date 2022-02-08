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

import pytest
import ray
import torch

from ludwig.api import LudwigModel
from ludwig.backend import LOCAL_BACKEND
from ludwig.backend.ray import get_trainer_kwargs, RayBackend
from ludwig.constants import TRAINER
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


def run_api_experiment(config, data_parquet):
    # Sanity check that we get 4 slots over 1 host
    kwargs = get_trainer_kwargs()
    if torch.cuda.device_count() > 0:
        assert kwargs.get("num_workers") == torch.cuda.device_count(), kwargs
        assert kwargs.get("resources_per_worker") is None, kwargs
        assert kwargs.get("use_gpu"), kwargs
    else:
        assert kwargs.get("num_workers") == 1, kwargs
        assert kwargs.get("resources_per_worker").get("CPU") == 2, kwargs
        assert not kwargs.get("use_gpu"), kwargs

    # Train on Parquet
    model = train_with_backend(RAY_BACKEND_CONFIG, config, dataset=data_parquet, evaluate=False)

    assert isinstance(model.backend, RayBackend)
    assert model.backend.df_engine.parallelism == RAY_BACKEND_CONFIG["processor"]["parallelism"]


def run_split_api_experiment(config, data_parquet):
    train_fname, val_fname, test_fname = split(data_parquet)

    # Train
    train_with_backend(RAY_BACKEND_CONFIG, config, training_set=train_fname, evaluate=False, predict=False)

    # Train + Validation
    train_with_backend(
        RAY_BACKEND_CONFIG, config, training_set=train_fname, validation_set=val_fname, evaluate=False, predict=False
    )

    # Train + Validation + Test
    train_with_backend(
        RAY_BACKEND_CONFIG,
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
def run_test_parquet(
    input_features,
    output_features,
    num_examples=100,
    run_fn=run_api_experiment,
    expect_error=False,
    num_cpus=2,
    num_gpus=None,
):
    with ray_start(num_cpus=num_cpus, num_gpus=num_gpus):
        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "output_size": 14},
            TRAINER: {"epochs": 2, "batch_size": 8},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_filename = os.path.join(tmpdir, "dataset.csv")
            dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=num_examples)
            dataset_parquet = create_data_set_to_use("parquet", dataset_csv)

            if expect_error:
                with pytest.raises(ValueError):
                    run_fn(config, data_parquet=dataset_parquet)
            else:
                run_fn(config, data_parquet=dataset_parquet)


@pytest.mark.distributed
def test_ray_tabular():
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
        binary_feature(),
        number_feature(normalization="zscore"),
    ]
    run_test_parquet(input_features, output_features)


@pytest.mark.skip(reason="TODO torch")
@pytest.mark.distributed
def test_ray_text():
    input_features = [
        text_feature(),
    ]
    output_features = [
        text_feature(reduce_input=None, decoder="tagger"),
    ]
    run_test_parquet(input_features, output_features)


@pytest.mark.skip(reason="TODO torch")
@pytest.mark.distributed
def test_ray_sequence():
    input_features = [sequence_feature(max_len=10, encoder="rnn", cell_type="lstm", reduce_output=None)]
    output_features = [sequence_feature(max_len=10, decoder="tagger", attention=False, reduce_input=None)]
    run_test_parquet(input_features, output_features)


@pytest.mark.distributed
def test_ray_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features = [audio_feature(folder=audio_dest_folder)]
        output_features = [binary_feature()]
        run_test_parquet(input_features, output_features)


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
        run_test_parquet(input_features, output_features)


@pytest.mark.skip(reason="flaky: ray is running out of resources")
@pytest.mark.distributed
def test_ray_split():
    input_features = [
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
    ]
    output_features = [category_feature(vocab_size=2, reduce_input="sum")]
    run_test_parquet(
        input_features,
        output_features,
        run_fn=run_split_api_experiment,
        num_cpus=4,
    )


@pytest.mark.distributed
def test_ray_timeseries():
    input_features = [timeseries_feature()]
    output_features = [number_feature()]
    run_test_parquet(input_features, output_features)


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
        run_test_parquet(input_features, output_features, expect_error=True)


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
        run_test_parquet(input_features, output_features, expect_error=True)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipIf(not torch.cuda.is_available(), reason="test requires gpu support")
@pytest.mark.distributed
def test_train_gpu_load_cpu():
    input_features = [
        category_feature(vocab_size=2, reduce_input="sum"),
        number_feature(normalization="zscore"),
    ]
    output_features = [
        binary_feature(),
    ]
    run_test_parquet(input_features, output_features, run_fn=_run_train_gpu_load_cpu, num_gpus=1)


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
