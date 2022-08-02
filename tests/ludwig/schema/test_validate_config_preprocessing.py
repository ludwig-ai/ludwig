import pytest
from jsonschema.exceptions import ValidationError

from ludwig.schema import validate_config
from tests.integration_tests.utils import binary_feature, category_feature


def test_config_preprocessing():
    input_features = [
        category_feature(),
        category_feature()
    ]
    output_features = [
        binary_feature()
    ]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "preprocessing": {
            "force_split": True,
            "split_probabilities": [0.6, 0.2, 0.2],
            "category": {
                "fill_value": "test",
            }
        }
    }

    validate_config(config)

    config["preprocessing"]["video"] = {"fill_value": "test"}

    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)

    del config["preprocessing"]["video"]
    config["preprocessing"]["number"] = {"most_common": 1000}

    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)

