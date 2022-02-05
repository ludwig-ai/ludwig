import contextlib
import os
import shutil

import pytest
import ray

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.contribs import WhyLogsCallback
from tests.integration_tests.utils import category_feature, generate_data, sequence_feature, spawn


@contextlib.contextmanager
def ray_start(num_cpus=2):
    res = ray.init(
        num_cpus=num_cpus,
        include_dashboard=False,
        object_store_memory=150 * 1024 * 1024,
    )
    try:
        yield res
    finally:
        ray.shutdown()


def test_whylogs_callback_local(tmpdir):
    epochs = 2
    batch_size = 8
    num_examples = 32

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": epochs, "batch_size": batch_size},
    }

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "train.csv"), num_examples=num_examples
    )
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    exp_name = "whylogs_test_local"
    callback = WhyLogsCallback()

    model = LudwigModel(config, callbacks=[callback])
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, experiment_name=exp_name)
    _, _ = model.predict(test_csv)

    local_training_output_dir = "output/training"
    local_prediction_output_dir = "output/prediction"

    assert os.path.isdir(local_training_output_dir) is True
    assert os.path.isdir(local_prediction_output_dir) is True


@pytest.mark.distributed
def test_whylogs_callback_dask(tmpdir):
    num_examples = 100

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "train.csv"), num_examples=num_examples
    )
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    run_dask(input_features, output_features, data_csv, val_csv, test_csv)
    local_training_output_dir = "output/training"
    local_prediction_output_dir = "output/prediction"

    assert os.path.isdir(local_training_output_dir) is True
    assert os.path.isdir(local_prediction_output_dir) is True


@spawn
def run_dask(input_features, output_features, data_csv, val_csv, test_csv):
    epochs = 2
    batch_size = 8
    backend = {
        "type": "ray",
        "processor": {
            "parallelism": 2,
        },
        "trainer": {
            "use_gpu": False,
            "num_workers": 2,
            "resources_per_worker": {
                "CPU": 1,
                "GPU": 0,
            },
        },
    }
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": epochs, "batch_size": batch_size},
    }

    with ray_start(num_cpus=4):
        exp_name = "whylogs_test_ray"
        callback = WhyLogsCallback()
        model = LudwigModel(config, backend=backend, callbacks=[callback])
        model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, experiment_name=exp_name)
        _, _ = model.predict(test_csv)
