import os
from tempfile import TemporaryDirectory

import pytest
import yaml

from ludwig.constants import (
    ACTIVE,
    CLIP,
    COLUMN,
    COMBINER,
    DECODER,
    DEFAULT_VALIDATION_METRIC,
    DEFAULTS,
    DEPENDENCIES,
    ENCODER,
    HYPEROPT,
    INPUT_FEATURES,
    INPUT_SIZE,
    LOSS,
    NAME,
    NUM_CLASSES,
    OPTIMIZER,
    OUTPUT_FEATURES,
    PREPROCESSING,
    PROC_COLUMN,
    REDUCE_DEPENDENCIES,
    REDUCE_INPUT,
    TIED,
    TRAINER,
    TYPE,
)
from ludwig.schema.model_config import ModelConfig
from ludwig.schema.utils import BaseMarshmallowConfig, convert_submodules

config_sections = {INPUT_FEATURES, OUTPUT_FEATURES, PREPROCESSING, TRAINER, COMBINER, DEFAULTS, HYPEROPT}


def test_config_object():
    config = {
        "input_features": [
            {
                "name": "text_feature",
                "type": "text",
                "preprocessing": {
                    "missing_value_strategy": "drop_row",
                },
                "encoder": {
                    "type": "rnn",
                    "bidirectional": True,
                    "representation": "dense",
                    "num_layers": 2,
                },
            },
            {
                "name": "image_feature_1",
                "type": "image",
                "preprocessing": {
                    "height": 7,
                    "width": 7,
                    "num_channels": 4,
                },
                "encoder": {
                    "type": "resnet",
                    "num_channels": 4,
                    "dropout": 0.1,
                    "resnet_size": 100,
                },
            },
        ],
        "output_features": [
            {
                "name": "category_feature",
                "type": "category",
                "top_k": 3,
                "preprocessing": {
                    "missing_value_strategy": "bfill",
                },
                "decoder": {
                    "type": "classifier",
                    "num_classes": 10,
                    "use_bias": False,
                },
            },
        ],
        "combiner": {
            "type": "concat",
            "output_size": 512,
            "weights_initializer": "xavier_uniform",
            "dropout": 0.2,
        },
        "trainer": {
            "epochs": 50,
            "batch_size": "auto",
            "optimizer": {
                "type": "adam",
                "beta1": 0.8,
                "beta2": 0.999,
                "epsilon": 5e-09,
            },
        },
    }

    config_object = ModelConfig.from_dict(config)
    assert config_object.input_features.text_feature.encoder.type == "rnn"
    assert config_object.input_features.text_feature.encoder.num_layers == 2
    assert config_object.input_features.text_feature.preprocessing.missing_value_strategy == "drop_row"

    assert config_object.defaults.text.encoder.type != "rnn"
    assert config_object.defaults.text.preprocessing.missing_value_strategy != "drop_row"

    assert config_object.output_features.category_feature.decoder.num_classes == 10
    assert config_object.output_features.category_feature.top_k == 3

    assert config_object.combiner.output_size == 512
    assert config_object.combiner.weights_initializer == "xavier_uniform"
    assert config_object.combiner.fc_layers is None

    assert config_object.trainer.epochs == 50
    assert config_object.trainer.batch_size == "auto"

    assert config_object.trainer.optimizer.type == "adam"
    assert config_object.trainer.optimizer.beta1 == 0.8
    assert config_object.trainer.optimizer.beta2 == 0.999
    assert config_object.trainer.optimizer.epsilon == 5e-09


def test_config_object_defaults():
    config = {
        "input_features": [
            {"name": "number_feature", "type": "number"},
            {
                "name": "text_feature_1",
                "type": "text",
                "encoder": {
                    "type": "rnn",
                    "activation": "sigmoid",
                },
            },
            {
                "name": "text_feature_2",
                "type": "text",
            },
        ],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
        "defaults": {
            "number": {"preprocessing": {"missing_value_strategy": "drop_row"}, "encoder": {"type": "dense"}},
            "text": {
                "preprocessing": {
                    "missing_value_strategy": "drop_row",
                },
                "encoder": {
                    "type": "stacked_parallel_cnn",
                    "activation": "tanh",
                },
            },
        },
    }

    config_object = ModelConfig.from_dict(config)
    assert config_object.input_features.number_feature.preprocessing.missing_value_strategy == "drop_row"
    assert config_object.input_features.number_feature.encoder.type == "dense"

    assert config_object.input_features.text_feature_1.encoder.type == "rnn"
    assert config_object.input_features.text_feature_1.encoder.activation == "sigmoid"
    assert config_object.input_features.text_feature_1.preprocessing.missing_value_strategy == "drop_row"

    assert config_object.input_features.text_feature_2.encoder.type == "stacked_parallel_cnn"
    assert config_object.input_features.text_feature_2.encoder.activation == "tanh"
    assert config_object.input_features.text_feature_2.preprocessing.missing_value_strategy == "drop_row"


def test_config_object_to_config_dict():
    config = {
        "input_features": [
            {"name": "number_feature", "type": "number"},
        ],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
    }

    config_object = ModelConfig.from_dict(config)
    config_dict = config_object.to_dict()

    for section in config_sections:
        assert section in config_dict
    assert len(config_dict[DEFAULTS]) == 13
    assert set(config_dict[INPUT_FEATURES][0].keys()) == {
        NAME,
        ACTIVE,
        TYPE,
        COLUMN,
        PROC_COLUMN,
        TIED,
        PREPROCESSING,
        ENCODER,
    }
    assert set(config_dict[OUTPUT_FEATURES][0].keys()) == {
        NAME,
        ACTIVE,
        TYPE,
        COLUMN,
        PROC_COLUMN,
        PREPROCESSING,
        DECODER,
        LOSS,
        REDUCE_INPUT,
        DEPENDENCIES,
        INPUT_SIZE,
        CLIP,
        REDUCE_DEPENDENCIES,
        NUM_CLASSES,
        DEFAULT_VALIDATION_METRIC,
    }


def test_update_config_object():
    config = {
        "input_features": [
            {"name": "text_feature", "type": "text"},
        ],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
    }

    config_object = ModelConfig.from_dict(config)

    assert config_object.input_features.text_feature.encoder.type == "parallel_cnn"
    assert config_object.input_features.text_feature.encoder.max_sequence_length is None

    temp_config = {
        "input_features": [
            {"name": "text_feature", "type": "text", "encoder": {"type": "parallel_cnn", "max_sequence_length": 10}},
        ],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
    }

    config_object.update_with_dict(temp_config)

    assert config_object.input_features.text_feature.encoder.max_sequence_length == 10


def test_constructors_yaml():
    config = {
        "input_features": [
            {"name": "text_feature", "type": "text", "encoder": {"type": "parallel_cnn", "max_sequence_length": 10}},
        ],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
    }

    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.yaml")
        with open(file_path, "w") as file:
            yaml.dump(config, file)

        config_obj = ModelConfig.from_yaml(file_path)

    for section in config_sections:
        assert hasattr(config_obj, section)


def test_constructors_dict():
    config = {
        "input_features": [
            {"name": "text_feature", "type": "text", "encoder": {"type": "parallel_cnn", "max_sequence_length": 10}},
        ],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
    }

    config_obj = ModelConfig.from_dict(config)

    for section in config_sections:
        assert hasattr(config_obj, section)


def test_feature_enabling_disabling():
    config = {
        "input_features": [{"name": "text_feature", "type": "text"}, {"name": "category_feature", "type": "number"}],
        "output_features": [
            {
                "name": "number_output_feature",
                "type": "number",
            },
        ],
    }

    config_obj = ModelConfig.from_dict(config)

    assert config_obj.input_features.text_feature.active
    assert config_obj.input_features.category_feature.active

    config_obj.input_features.text_feature.disable()

    assert not config_obj.input_features.text_feature.active


def test_sequence_combiner():
    config = {
        "input_features": [{"name": "text_feature", "type": "text"}],
        "output_features": [{"name": "number_output_feature", "type": "number"}],
        "combiner": {"type": "sequence", "encoder": {"type": "rnn"}},
    }

    config_obj = ModelConfig.from_dict(config)

    assert config_obj.combiner.type == "sequence"
    assert config_obj.combiner.encoder.type == "rnn"
    assert config_obj.combiner.encoder.cell_type == "rnn"


@pytest.mark.parametrize(
    "session",
    [
        {"sess_id": 0, "encoder": "parallel_cnn", "loss": {"type": "mean_squared_error"}},
        {"sess_id": 1, "encoder": "cnnrnn", "loss": {"type": "mean_absolute_error"}},
        {"sess_id": 2, "encoder": "parallel_cnn", "loss": {"type": "mean_absolute_error"}},
    ],
)
def test_shared_state(session):
    config = {
        "input_features": [
            {"name": "text_feature", "type": "text", "encoder": {"type": session["encoder"]}},
            {"name": "text_feature_2", "type": "text"},
        ],
        "output_features": [
            {"name": "number_output_feature", "type": "number"},
            {"name": "category_feature", "type": "category", "preprocessing": {"missing_value_strategy": "bfill"}},
        ],
        "defaults": {"text": {"encoder": {"type": session["encoder"]}}},
    }

    if session["sess_id"] == 1:
        del config[OUTPUT_FEATURES][1]["preprocessing"]

    if session["sess_id"] == 2:
        del config[INPUT_FEATURES][0]["encoder"]
        del config[DEFAULTS]

    config_obj = ModelConfig.from_dict(config)

    if session["sess_id"] == 0:
        config_obj.input_features.text_feature.encoder.max_sequence_length = 10
        config_obj.input_features.text_feature.tied = "text_feature_2"

        assert config_obj.defaults.text.encoder.max_sequence_length is None  # Test no link w/ defaults config
        assert config_obj.input_features.text_feature.tied == "text_feature_2"  # Test tied set as expected

    if session["sess_id"] == 1:
        config_obj.output_features.number_output_feature.loss.weight = 2.0

        # Test previous edits to config don't carry over
        assert config_obj.output_features.category_feature.preprocessing.missing_value_strategy == "drop_row"
        assert config_obj.defaults.text.encoder.max_sequence_length is None  # Test no link w/ previous encoder config
        assert config_obj.input_features.text_feature.tied is None  # Test no link w/ previous text feature config
        assert config_obj.output_features.number_output_feature.loss.weight == 2.0  # Test loss weight set as expected

    if session["sess_id"] == 2:
        assert config_obj.input_features.text_feature.encoder.type == "parallel_cnn"
        assert config_obj.output_features.number_output_feature.loss.weight == 1.0  # Test no link previous loss config
        assert config_obj.defaults.text.encoder.max_sequence_length is None  # Test no link w/ first encoder config
        assert config_obj.input_features.text_feature.tied is None  # Test no link w/ first tied setting


def test_convert_submodules():
    config = {
        "input_features": [
            {"name": "text_feature", "type": "text"},
        ],
        "output_features": [{"name": "number_output_feature", "type": "number"}],
    }

    config_obj = ModelConfig.from_dict(config)
    trainer = convert_submodules(config_obj.trainer.__dict__)
    input_features = list(convert_submodules(config_obj.input_features.__dict__).values())

    assert not isinstance(trainer[OPTIMIZER], BaseMarshmallowConfig)
    assert not isinstance(input_features[0][PREPROCESSING], BaseMarshmallowConfig)


def test_defaults_mixins():
    config = {
        "input_features": [
            {"name": "text_feature", "type": "text"},
        ],
        "output_features": [{"name": "number_output_feature", "type": "number"}],
    }

    config_obj = ModelConfig.from_dict(config)

    assert config_obj.defaults.audio.to_dict().keys() == {ENCODER, PREPROCESSING}
    assert config_obj.defaults.category.to_dict().keys() == {ENCODER, PREPROCESSING, DECODER, LOSS}
