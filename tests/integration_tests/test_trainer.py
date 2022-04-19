import os
import shutil
import tempfile
from unittest import mock

import pytest
import ray

from ludwig.api import LudwigModel
from ludwig.backend.horovod import HorovodBackend
from ludwig.callbacks import Callback
from ludwig.constants import BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, TRAINER
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    LocalTestBackend,
    ray_cluster,
    sequence_feature,
)


@pytest.fixture(scope="module")
def ray_test_cluster():
    with ray_cluster():
        yield


def test_tune_batch_size_and_lr(tmpdir):
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [sequence_feature(reduce_output="sum")]
        output_features = [category_feature(vocab_size=2, reduce_input="sum")]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "output_size": 14},
            TRAINER: {
                "epochs": 2,
                "batch_size": "auto",
                "eval_batch_size": "auto",
                "learning_rate": "auto",
            },
        }

        model = LudwigModel(config, backend=LocalTestBackend())

        # check preconditions
        assert model.config[TRAINER][BATCH_SIZE] == "auto"
        assert model.config[TRAINER][EVAL_BATCH_SIZE] == "auto"
        assert model.config[TRAINER][LEARNING_RATE] == "auto"

        _, _, output_directory = model.train(
            training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=outdir
        )

        def check_postconditions(model):
            # check batch size
            assert model.config[TRAINER][BATCH_SIZE] != "auto"
            assert model.config[TRAINER][BATCH_SIZE] > 1

            assert model.config[TRAINER][EVAL_BATCH_SIZE] != "auto"
            assert model.config[TRAINER][EVAL_BATCH_SIZE] > 1

            assert model.config[TRAINER][BATCH_SIZE] == model.config[TRAINER][EVAL_BATCH_SIZE]

            # check learning rate
            assert model.config[TRAINER][LEARNING_RATE] != "auto"
            assert model.config[TRAINER][LEARNING_RATE] > 0

        check_postconditions(model)

        model = LudwigModel.load(os.path.join(output_directory, "model"))

        # loaded model should retain the tuned params
        check_postconditions(model)


@ray.remote
def run_scale_lr(config, data_csv, num_workers, outdir):
    class FakeHorovodBackend(HorovodBackend):
        def initialize(self):
            import horovod.torch as hvd

            hvd.init()

            self._horovod = mock.Mock(wraps=hvd)
            self._horovod.size.return_value = num_workers

    class TestCallback(Callback):
        def __init__(self):
            self.lr = None

        def on_trainer_train_teardown(self, trainer, progress_tracker, save_path, is_coordinator: bool):
            for g in trainer.optimizer.param_groups:
                self.lr = g["lr"]

    callback = TestCallback()
    model = LudwigModel(config, backend=FakeHorovodBackend(), callbacks=[callback])
    model.train(dataset=data_csv, output_directory=outdir)
    return callback.lr


@pytest.mark.parametrize("learning_rate_scaling, expected_lr", [("constant", 1), ("sqrt", 2), ("linear", 4)])
def test_scale_lr(learning_rate_scaling, expected_lr, tmpdir, ray_test_cluster):
    base_lr = 1.0
    num_workers = 4

    outdir = os.path.join(tmpdir, "output")

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {
            "epochs": 2,
            "learning_rate": base_lr,
            "learning_rate_scaling": learning_rate_scaling,
        },
    }

    actual_lr = ray.get(run_scale_lr.remote(config, data_csv, num_workers, outdir))
    assert actual_lr == expected_lr
