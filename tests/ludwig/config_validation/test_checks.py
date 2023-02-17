"""Tests for interdependent parameters.

Note that all testing should be done with the public API, rather than individual checks.

```
ModelConfig.from_dict(config)
```
"""

import pytest

from ludwig.error import ConfigValidationError
from ludwig.schema.model_types.base import ModelConfig
from tests.integration_tests.utils import binary_feature, text_feature


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
        ModelConfig.from_dict(config)


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
        ModelConfig.from_dict(config)


def test_unsupported_features_config():
    # GBMs don't support text features.
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(
            {
                "input_features": [text_feature()],
                "output_features": [binary_feature()],
                "model_type": "gbm",
            }
        )

    # GBMs don't support output text features.
    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(
            {
                "input_features": [binary_feature()],
                "output_features": [text_feature()],
                "model_type": "gbm",
            }
        )

    # ECD supports output text features.
    ModelConfig.from_dict(
        {
            "input_features": [binary_feature()],
            "output_features": [text_feature()],
            "model_type": "ecd",
        }
    )
