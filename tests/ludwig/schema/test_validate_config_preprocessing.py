import pytest

from ludwig.error import ConfigValidationError
from ludwig.schema.model_config import ModelConfig
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

    ModelConfig(config)

    # TODO(ksbrar): Circle back after discussing whether additional properties should be allowed long-term.
    # config["preprocessing"]["fake_parameter"] = True

    # with pytest.raises(Exception):
    #     ModelConfig(config)


def test_balance_non_binary_failure():
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "number"},
            {"name": "random_1", "proc_column": "random_1", "type": "number"},
            {"name": "random_2", "proc_column": "random_2", "type": "number"},
        ],
        "output_features": [{"name": "Label", "proc_column": "Label", "type": "number"}],
        "preprocessing": {"oversample_minority": 0.2},
    }

    with pytest.raises(ConfigValidationError):
        ModelConfig(config)


def test_balance_multiple_class_failure():
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "number"},
            {"name": "random_1", "proc_column": "random_1", "type": "number"},
            {"name": "random_2", "proc_column": "random_2", "type": "number"},
        ],
        "output_features": [
            {"name": "Label", "proc_column": "Label", "type": "binary"},
            {"name": "Label2", "proc_column": "Label2", "type": "binary"},
        ],
        "preprocessing": {"oversample_minority": 0.2},
    }

    with pytest.raises(ConfigValidationError):
        ModelConfig(config)
