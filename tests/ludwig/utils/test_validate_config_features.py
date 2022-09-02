import pytest
from jsonschema.exceptions import ValidationError

from ludwig.schema import validate_config
from tests.integration_tests.utils import binary_feature, category_feature, number_feature, text_feature


def test_config_input_output_features():
    config = {
        "input_features": [
            category_feature(encoder={"type": "dense"}),
            number_feature(encoder={"type": "passthrough"}),
        ],
        "output_features": [binary_feature(decoder={"type": "regressor"})],
    }

    validate_config(config)


def test_incorrect_input_features_config():
    config = {
        "input_features": [
            category_feature(preprocessing={"normalization": "zscore"}),
        ],
        "output_features": [binary_feature()],
    }

    # Not a preprocessing param for category feature
    with pytest.raises(ValidationError):
        validate_config(config)

    config = {
        "input_features": [
            text_feature(preprocessing={"padding_symbol": 0}),
        ],
        "output_features": [binary_feature()],
    }

    # Incorrect type for padding_symbol preprocessing param
    with pytest.raises(ValidationError):
        validate_config(config)

    config = {
        "input_features": [
            binary_feature(),
        ],
        "output_features": [binary_feature()],
    }
    del config["input_features"][0]["type"]

    # Incorrect type for padding_symbol preprocessing param
    with pytest.raises(ValidationError):
        validate_config(config)


def test_incorrect_output_features_config():
    config = {
        "input_features": [
            number_feature(),
        ],
        "output_features": [binary_feature(decoder="classifier")],
    }

    # Invalid decoder for binary output feature
    with pytest.raises(ValidationError):
        validate_config(config)
