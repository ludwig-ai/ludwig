from ludwig.constants import INPUT_FEATURES, OUTPUT_FEATURES
from ludwig.utils.backward_compatibility import (
    _upgrade_encoder_decoder_params,
    _upgrade_feature,
    _upgrade_preprocessing,
)


def test_preprocessing_backward_compatibility():
    # From v0.5.3.
    preprocessing_config = {
        "force_split": False,
        "split_probabilities": [0.7, 0.1, 0.2],
        "stratify": None,
    }

    _upgrade_preprocessing(preprocessing_config)

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
    _upgrade_preprocessing(global_preprocessing_config)

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
                "top_k": 3,
                "preprocessing": {
                    "missing_value_strategy": "backfill",
                },
                "decoder": {
                    "type": "classifier",
                    "num_classes": 10,
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
                    "use_bias": True,
                    "bias_initializer": "constant",
                },
            },
        ],
    }
