import pytest

from ludwig.schema import validate_config
from tests.integration_tests.utils import binary_feature, category_feature


def test_config_preprocessing():
    input_features = [category_feature(), category_feature()]
    output_features = [binary_feature()]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "preprocessing": {
            "split": {
                "type": "random",
                "probabilities": [0.6, 0.2, 0.2],
            },
            "oversample_minority": 0.4,
        },
    }

    validate_config(config)

    config["preprocessing"]["fake_parameter"] = True

    with pytest.raises(Exception):
        validate_config(config)
