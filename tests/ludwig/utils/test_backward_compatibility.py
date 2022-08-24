import pytest

from ludwig.constants import (
    EVAL_BATCH_SIZE,
    HYPEROPT,
    INPUT_FEATURES,
    NUMBER,
    OUTPUT_FEATURES,
    PREPROCESSING,
    SPLIT,
    TRAINER,
    TYPE,
)
from ludwig.utils.backward_compatibility import (
    _upgrade_encoder_decoder_params,
    _upgrade_feature,
    _upgrade_preprocessing_split,
    upgrade_to_latest_version,
)


def test_preprocessing_backward_compatibility():
    # From v0.5.3.
    preprocessing_config = {
        "force_split": False,
        "split_probabilities": [0.7, 0.1, 0.2],
        "stratify": None,
    }

    _upgrade_preprocessing_split(preprocessing_config)

    assert preprocessing_config == {
        "split": {"probabilities": [0.7, 0.1, 0.2], "type": "random"},
    }


def test_audio_feature_backward_compatibility():
    # From v0.5.3.

    audio_feature_preprocessing_config = {
        "name": "audio_feature",
        "type": "audio",
        "preprocessing": {
            "audio_file_length_limit_in_s": 7.5,
            "missing_value_strategy": "backfill",
            "in_memory": True,
            "padding_value": 0,
            "norm": None,
            "audio_feature": {
                "type": "fbank",
                "window_length_in_s": 0.04,
                "window_shift_in_s": 0.02,
                "num_fft_points": None,
                "window_type": "hamming",
                "num_filter_bands": 80,
            },
        },
    }

    global_preprocessing_config = {
        "audio": {
            "audio_file_length_limit_in_s": 7.5,
            "missing_value_strategy": "backfill",
            "in_memory": True,
            "padding_value": 0,
            "norm": None,
            "audio_feature": {
                "type": "fbank",
                "window_length_in_s": 0.04,
                "window_shift_in_s": 0.02,
                "num_fft_points": None,
                "window_type": "hamming",
                "num_filter_bands": 80,
            },
        },
    }

    _upgrade_feature(audio_feature_preprocessing_config)
    _upgrade_preprocessing_split(global_preprocessing_config)

    assert global_preprocessing_config == {
        "audio": {
            "audio_file_length_limit_in_s": 7.5,
            "missing_value_strategy": "backfill",
            "in_memory": True,
            "padding_value": 0,
            "norm": None,
            "type": "fbank",
            "window_length_in_s": 0.04,
            "window_shift_in_s": 0.02,
            "num_fft_points": None,
            "window_type": "hamming",
            "num_filter_bands": 80,
        }
    }

    assert audio_feature_preprocessing_config == {
        "name": "audio_feature",
        "type": "audio",
        "preprocessing": {
            "audio_file_length_limit_in_s": 7.5,
            "missing_value_strategy": "backfill",
            "in_memory": True,
            "padding_value": 0,
            "norm": None,
            "type": "fbank",
            "window_length_in_s": 0.04,
            "window_shift_in_s": 0.02,
            "num_fft_points": None,
            "window_type": "hamming",
            "num_filter_bands": 80,
        },
    }


def test_encoder_decoder_backwards_compatibility():
    old_config = {
        "input_features": [
            {
                "name": "text_feature",
                "type": "text",
                "preprocessing": {
                    "missing_value_strategy": "drop_row",
                },
                "encoder": "rnn",
                "bidirectional": True,
                "representation": "dense",
                "num_layers": 2,
            },
            {
                "name": "image_feature_1",
                "type": "image",
                "preprocessing": {
                    "height": 7.5,
                    "width": 7.5,
                    "num_channels": 4,
                },
                "encoder": "resnet",
                "num_channels": 4,
                "dropout": 0.1,
                "resnet_size": 100,
            },
            {
                "name": "image_feature_2",
                "type": "image",
                "tied": "image_feature_1",
                "preprocessing": {
                    "height": 7.5,
                    "width": 7.5,
                    "num_channels": 4,
                },
                "encoder": "resnet",
            },
        ],
        "output_features": [
            {
                "name": "category_feature",
                "type": "category",
                "top_k": 3,
                "preprocessing": {
                    "missing_value_strategy": "backfill",
                },
                "decoder": "classifier",
                "num_classes": 10,
                "use_bias": False,
            },
            {
                "name": "binary_feature",
                "type": "binary",
                "dependencies": ["category_feature"],
                "loss": {
                    "type": "cross_entropy",
                },
                "reduce_dependencies": "mean",
                "decoder": "regressor",
                "use_bias": True,
                "bias_initializer": "constant",
            },
            {
                "name": "vector_feature",
                "type": "vector",
                "decoder": "projector",
                "num_fc_layers": 5,
                "output_size": 128,
                "activation": "tanh",
                "dropout": 0.1,
            },
        ],
    }

    for feature in old_config[INPUT_FEATURES]:
        _upgrade_encoder_decoder_params(feature, True)

    for feature in old_config[OUTPUT_FEATURES]:
        _upgrade_encoder_decoder_params(feature, False)

    assert old_config == {
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
                    "height": 7.5,
                    "width": 7.5,
                    "num_channels": 4,
                },
                "encoder": {
                    "type": "resnet",
                    "num_channels": 4,
                    "dropout": 0.1,
                    "resnet_size": 100,
                },
            },
            {
                "name": "image_feature_2",
                "type": "image",
                "tied": "image_feature_1",
                "preprocessing": {
                    "height": 7.5,
                    "width": 7.5,
                    "num_channels": 4,
                },
                "encoder": {"type": "resnet"},
            },
        ],
        "output_features": [
            {
                "name": "category_feature",
                "type": "category",
                "num_classes": 10,
                "top_k": 3,
                "preprocessing": {
                    "missing_value_strategy": "backfill",
                },
                "decoder": {
                    "type": "classifier",
                    "fc_use_bias": False,
                    "use_bias": False,
                },
            },
            {
                "name": "binary_feature",
                "type": "binary",
                "dependencies": ["category_feature"],
                "loss": {
                    "type": "cross_entropy",
                },
                "reduce_dependencies": "mean",
                "decoder": {
                    "type": "regressor",
                    "fc_use_bias": True,
                    "fc_bias_initializer": "constant",
                    "bias_initializer": "constant",
                    "use_bias": True,
                },
            },
            {
                "name": "vector_feature",
                "type": "vector",
                "decoder": {
                    "type": "projector",
                    "num_fc_layers": 5,
                    "fc_output_size": 128,
                    "fc_activation": "tanh",
                    "fc_dropout": 0.1,
                    "output_size": 128,
                    "activation": "tanh",
                    "dropout": 0.1,
                },
            },
        ],
    }


def test_deprecated_field_aliases():
    config = {
        "ludwig_version": "0.4",
        INPUT_FEATURES: [{"name": "num_in", "type": "numerical"}],
        OUTPUT_FEATURES: [{"name": "num_out", "type": "numerical"}],
        "training": {
            "epochs": 2,
            "eval_batch_size": 0,
        },
        HYPEROPT: {
            "parameters": {
                "training.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
            },
            "goal": "minimize",
            "sampler": {"type": "grid", "num_samples": 2, "scheduler": {"type": "fifo"}},
            "executor": {
                "type": "grid",
                "search_alg": "bohb",
            },
        },
    }

    updated_config = upgrade_to_latest_version(config)

    assert updated_config["input_features"][0][TYPE] == NUMBER
    assert updated_config["output_features"][0][TYPE] == NUMBER

    assert "training" not in updated_config
    assert updated_config[TRAINER]["epochs"] == 2
    assert updated_config[TRAINER][EVAL_BATCH_SIZE] is None

    hparams = updated_config[HYPEROPT]["parameters"]
    assert "training.learning_rate" not in hparams
    assert "trainer.learning_rate" in hparams

    assert "sampler" not in updated_config[HYPEROPT]

    assert updated_config[HYPEROPT]["executor"]["type"] == "ray"
    assert "num_samples" in updated_config[HYPEROPT]["executor"]
    assert "scheduler" in updated_config[HYPEROPT]["executor"]


@pytest.mark.parametrize("force_split", [None, False, True])
@pytest.mark.parametrize("stratify", [None, "cat_in"])
def test_deprecated_split_aliases(stratify, force_split):
    split_probabilities = [0.6, 0.2, 0.2]
    config = {
        "ludwig_version": "0.4",
        INPUT_FEATURES: [{"name": "num_in", "type": "number"}, {"name": "cat_in", "type": "category"}],
        OUTPUT_FEATURES: [{"name": "num_out", "type": "number"}],
        PREPROCESSING: {
            "force_split": force_split,
            "split_probabilities": split_probabilities,
            "stratify": stratify,
        },
    }

    updated_config = upgrade_to_latest_version(config)

    assert "force_split" not in updated_config[PREPROCESSING]
    assert "split_probabilities" not in updated_config[PREPROCESSING]
    assert "stratify" not in updated_config[PREPROCESSING]

    assert SPLIT in updated_config[PREPROCESSING]
    split = updated_config[PREPROCESSING][SPLIT]

    assert split["probabilities"] == split_probabilities
    if stratify is None:
        if force_split:
            assert split.get(TYPE) == "random"
    else:
        assert split.get(TYPE) == "stratify"
        assert split.get("column") == stratify
