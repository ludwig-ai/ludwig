from ludwig.utils.backward_compatibility import _upgrade_feature, _upgrade_preprocessing_split


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
