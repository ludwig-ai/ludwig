from ludwig.utils.backward_compatibility import _upgrade_preprocessing


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
