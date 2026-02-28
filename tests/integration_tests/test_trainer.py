import logging
import os
import shutil
from unittest import mock

import pytest
import torch
from packaging.version import parse as parse_version

from ludwig.api import LudwigModel
from ludwig.constants import (
    BATCH_SIZE,
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    INPUT_FEATURES,
    MAX_BATCH_SIZE_DATASET_FRACTION,
    OUTPUT_FEATURES,
    TRAINER,
)
from ludwig.globals import MODEL_FILE_NAME
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    text_feature,
    vector_feature,
)


def test_tune_learning_rate(tmpdir):
    config = {
        INPUT_FEATURES: [text_feature(), binary_feature()],
        OUTPUT_FEATURES: [binary_feature()],
        TRAINER: {
            "train_steps": 1,
            BATCH_SIZE: 128,
            "learning_rate": "auto",
        },
    }

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(config[INPUT_FEATURES], config[OUTPUT_FEATURES], csv_filename)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)

    assert model.config_obj.trainer.learning_rate == 0.0001


@pytest.mark.parametrize(
    "is_cpu,effective_batch_size,eval_batch_size",
    [
        (True, "auto", "auto"),
        (False, 256, 128),
        (True, "auto", None),
    ],
    ids=["cpu_auto", "gpu_fixed", "cpu_no_eval_bs"],
)
def test_ecd_tune_batch_size_and_lr(tmpdir, eval_batch_size, effective_batch_size, is_cpu):
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
        EPOCHS: 2,
        EFFECTIVE_BATCH_SIZE: effective_batch_size,
        BATCH_SIZE: "auto",
        "gradient_accumulation_steps": "auto",
        "learning_rate": "auto",
    }

    if eval_batch_size:
        trainer[EVAL_BATCH_SIZE] = eval_batch_size

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: trainer,
    }

    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)

    # check preconditions
    assert model.config_obj.trainer.effective_batch_size == effective_batch_size
    assert model.config_obj.trainer.batch_size == "auto"
    assert model.config_obj.trainer.gradient_accumulation_steps == "auto"
    assert model.config_obj.trainer.eval_batch_size == eval_batch_size
    assert model.config_obj.trainer.learning_rate == "auto"

    with mock.patch("ludwig.trainers.trainer.Trainer.is_cpu_training") as mock_fn:
        mock_fn.return_value = is_cpu
        _, _, output_directory = model.train(
            training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir
        )

    def check_postconditions(model):
        # check batch size
        assert model.config_obj.trainer.effective_batch_size == effective_batch_size
        assert model.config_obj.trainer.batch_size != "auto"
        assert model.config_obj.trainer.batch_size > 1

        # check gradient accumulation
        assert model.config_obj.trainer.gradient_accumulation_steps != "auto"
        if effective_batch_size == "auto":
            assert model.config_obj.trainer.gradient_accumulation_steps == 1
        else:
            batch_size = model.config_obj.trainer.batch_size
            assert model.config_obj.trainer.gradient_accumulation_steps == effective_batch_size // batch_size

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

    model = LudwigModel.load(os.path.join(output_directory, MODEL_FILE_NAME))

    # loaded model should retain the tuned params
    check_postconditions(model)


def test_changing_parameters_on_plateau(tmpdir):
    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {
            EPOCHS: 2,
            BATCH_SIZE: 128,
            "learning_rate": 1.0,
            "reduce_learning_rate_on_plateau": 1,
            "increase_batch_size_on_plateau": 1,
        },
    }
    model = LudwigModel(config, backend=LocalTestBackend())

    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)


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
        EPOCHS: 2,
        "use_mixed_precision": True,
    }

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: trainer,
    }

    # Just test that training completes without error.
    # TODO(travis): We may want to expand upon this in the future to include some checks on model
    # convergence like gradient magnitudes, etc. Should also add distributed tests.
    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)


@pytest.mark.skipif(
    parse_version(torch.__version__) < parse_version("2.0"), reason="Model compilation requires PyTorch >= 2.0"
)
def test_compile(tmpdir):
    input_features = [text_feature()]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    trainer = {
        EPOCHS: 2,
        "compile": True,
    }

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: trainer,
    }

    # Just test that training completes without error.
    # TODO(travis): We may want to expand upon this in the future to include some checks on model
    # convergence like gradient magnitudes, etc. Should also add distributed tests.
    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
def test_gradient_accumulation(gradient_accumulation_steps: int, tmpdir):
    input_features = [text_feature()]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename, num_examples=64)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    trainer = {
        EPOCHS: 2,
        BATCH_SIZE: 8,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: trainer,
    }

    # Just test that training completes without error.
    # TODO(travis): We may want to expand upon this in the future to include some checks on model
    # convergence like gradient magnitudes, etc. Should also add distributed tests.
    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)


def test_enable_gradient_checkpointing(tmpdir, caplog):
    """Test that gradient checkpointing is enabled when specified in the config and that it does not cause an error
    when the model does not have support for gradient checkpointing."""
    input_features = [text_feature()]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    data_csv = generate_data(input_features, output_features, csv_filename)
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {
            "train_steps": 2,
            BATCH_SIZE: 8,
            "enable_gradient_checkpointing": True,
        },
    }

    model = LudwigModel(config, backend=LocalTestBackend(), logging_level=logging.INFO)
    assert model.config_obj.trainer.enable_gradient_checkpointing

    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=tmpdir)

    # Check that the warning is emitted when the model does not support gradient checkpointing
    # but does not prevent training from starting.
    assert "Gradient checkpointing is currently only supported for model_type: llm. Skipping..." in caplog.text
