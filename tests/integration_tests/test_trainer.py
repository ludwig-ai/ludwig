import logging
import os
import shutil
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.constants import BATCH_SIZE, MAX_BATCH_SIZE_DATASET_FRACTION, TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
    RAY_BACKEND_CONFIG,
    sequence_feature,
    text_feature,
    vector_feature,
)

try:
    import dask
    import ray

    from ludwig.backend.horovod import HorovodBackend
    from ludwig.data.dataset.ray import RayDataset
    from ludwig.distributed.horovod import HorovodStrategy
    from ludwig.models.gbm import GBM
    from ludwig.schema.model_config import ModelConfig
    from ludwig.schema.trainer import GBMTrainerConfig
    from ludwig.trainers.trainer_lightgbm import LightGBMRayTrainer

    @ray.remote
    def run_scale_lr(config, data_csv, num_workers, outdir):
        class FakeHorovodBackend(HorovodBackend):
            def initialize(self):
                distributed = HorovodStrategy()
                self._distributed = mock.Mock(wraps=distributed)
                self._distributed.size.return_value = num_workers

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

except ImportError:
    dask = None
    ray = None


def test_tune_learning_rate(tmpdir):
    config = {
        "input_features": [text_feature(), binary_feature()],
        "output_features": [binary_feature()],
        TRAINER: {
            "train_steps": 1,
            BATCH_SIZE: 128,
            "learning_rate": "auto",
        },
    }

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(config["input_features"], config["output_features"], csv_filename)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)

    assert model.config_obj.trainer.learning_rate == 0.0001


@pytest.mark.parametrize("is_cpu", [True, False])
@pytest.mark.parametrize("eval_batch_size", ["auto", None, 128])
def test_tune_batch_size_and_lr(tmpdir, eval_batch_size, is_cpu):
    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [
        category_feature(decoder={"vocab_size": 2}, reduce_input="sum"),
        number_feature(),
        binary_feature(),
        vector_feature(),
    ]

    num_samples = 30
    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename, num_examples=num_samples)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    trainer = {
        "epochs": 2,
        "batch_size": "auto",
        "learning_rate": "auto",
    }

    if eval_batch_size:
        trainer["eval_batch_size"] = eval_batch_size

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: trainer,
    }

    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)

    # check preconditions
    assert model.config_obj.trainer.batch_size == "auto"
    assert model.config_obj.trainer.eval_batch_size == eval_batch_size
    assert model.config_obj.trainer.learning_rate == "auto"

    with mock.patch("ludwig.trainers.trainer.Trainer.is_cpu_training") as mock_fn:
        mock_fn.return_value = is_cpu
        _, _, output_directory = model.train(
            training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir
        )

    def check_postconditions(model):
        # check batch size
        assert model.config_obj.trainer.batch_size != "auto"
        assert model.config_obj.trainer.batch_size > 1

        # 4 is the largest possible batch size for this dataset (20% of dataset size)
        assert model.config_obj.trainer.batch_size <= MAX_BATCH_SIZE_DATASET_FRACTION * num_samples

        assert model.config_obj.trainer.eval_batch_size != "auto"
        assert model.config_obj.trainer.eval_batch_size > 1

        if eval_batch_size in ("auto", None):
            assert model.config_obj.trainer.batch_size == model.config_obj.trainer.eval_batch_size
        else:
            assert model.config_obj.trainer.eval_batch_size == eval_batch_size

        # check learning rate
        assert model.config_obj.trainer.learning_rate == 0.0001  # has sequence feature

    check_postconditions(model)

    model = LudwigModel.load(os.path.join(output_directory, "model"))

    # loaded model should retain the tuned params
    check_postconditions(model)


@pytest.mark.parametrize("learning_rate_scaling, expected_lr", [("constant", 1), ("sqrt", 2), ("linear", 4)])
@pytest.mark.distributed
def test_scale_lr(learning_rate_scaling, expected_lr, tmpdir, ray_cluster_2cpu):
    base_lr = 1.0
    num_workers = 4

    outdir = os.path.join(tmpdir, "output")

    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {
            "epochs": 2,
            BATCH_SIZE: 128,
            "learning_rate": base_lr,
            "learning_rate_scaling": learning_rate_scaling,
        },
    }

    actual_lr = ray.get(run_scale_lr.remote(config, data_csv, num_workers, outdir))
    assert actual_lr == expected_lr


def test_changing_parameters_on_plateau(tmpdir):
    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

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
            BATCH_SIZE: 128,
            "learning_rate": 1.0,
            "reduce_learning_rate_on_plateau": 1,
            "increase_batch_size_on_plateau": 1,
        },
    }
    model = LudwigModel(config, backend=LocalTestBackend())

    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)


@pytest.mark.distributed
def test_lightgbm_dataset_partition(ray_cluster_2cpu):
    # Create a LightGBM model with a Ray backend
    config = {
        "input_features": [{"name": "in_column", "type": "binary"}],
        "output_features": [{"name": "out_column", "type": "binary"}],
        "model_type": "gbm",
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        TRAINER: {"feature_pre_filter": False},
    }
    backend_config = {**RAY_BACKEND_CONFIG}
    backend_config["preprocessor_kwargs"] = {"num_cpu": 1}
    model = LudwigModel(config, backend=backend_config)
    lgbm_model = GBM(ModelConfig.from_dict(config))
    trainer = LightGBMRayTrainer(GBMTrainerConfig(), lgbm_model)

    def create_dataset(model: LudwigModel, size: int) -> RayDataset:
        df = pd.DataFrame(
            {
                "in_column_lm_J5T": np.random.randint(0, 1, size=(size,), dtype=np.uint8),
                "out_column_2Xl8CP": np.random.randint(0, 1, size=(size,), dtype=np.uint8),
            }
        )
        df = dask.dataframe.from_pandas(df, npartitions=1)
        return model.backend.dataset_manager.create(df, config=model.config, training_set_metadata={})

    # Create synthetic train, val, and test datasets with one block
    train_ds = create_dataset(model, int(1e4))
    val_ds = create_dataset(model, int(1e4))
    test_ds = create_dataset(model, int(1e4))

    # Test with no repartition. This occurs when the number of dataset blocks
    # is equal to the number of ray actors.
    trainer.ray_params.num_actors = 1
    trainer._construct_lgb_datasets(train_ds, validation_set=val_ds, test_set=test_ds)
    assert train_ds.ds.num_blocks() == 1
    assert val_ds.ds.num_blocks() == 1
    assert test_ds.ds.num_blocks() == 1

    # Test with repartition. This occurs when the number of dataset blocks
    # is less than the number of ray actors.
    trainer.ray_params.num_actors = 2
    trainer._construct_lgb_datasets(train_ds, validation_set=val_ds, test_set=test_ds)
    assert train_ds.ds.num_blocks() == 2
    assert val_ds.ds.num_blocks() == 2
    assert test_ds.ds.num_blocks() == 2

    # Test again with no repartition. This also occurs when the number of dataset blocks
    # is greater than the number of ray actors.
    trainer.ray_params.num_actors = 1
    trainer._construct_lgb_datasets(train_ds, validation_set=val_ds, test_set=test_ds)
    assert train_ds.ds.num_blocks() == 2
    assert val_ds.ds.num_blocks() == 2
    assert test_ds.ds.num_blocks() == 2


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
def test_mixed_precision(tmpdir):
    input_features = [text_feature()]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    trainer = {
        "epochs": 2,
        "use_mixed_precision": True,
    }

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: trainer,
    }

    # Just test that training completes without error.
    # TODO(travis): We may want to expand upon this in the future to include some checks on model
    # convergence like gradient magnitudes, etc. Should also add distributed tests.
    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)
