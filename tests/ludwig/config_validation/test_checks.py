"""Tests for interdependent parameters.

Note that all testing should be done with the public API, rather than individual checks.

```
ModelConfig.from_dict(config)
```
"""

import contextlib
from typing import Any, Dict, Optional

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


@pytest.mark.parametrize(
    "num_fc_layers,fc_layers,expect_success",
    [
        (None, None, True),
        (1, None, True),
        (None, [{"output_size": 256}], True),
        (0, [{"output_size": 256}], True),
        (0, None, False),
    ],
)
def test_comparator_fc_layer_config(
    num_fc_layers: Optional[int], fc_layers: Optional[Dict[str, Any]], expect_success: bool
):
    config = {
        "input_features": [
            {"name": "in1", "type": "category"},
            {"name": "in2", "type": "category"},
        ],
        "output_features": [
            {"name": "out1", "type": "binary"},
        ],
        "combiner": {
            "type": "comparator",
            "entity_1": ["in1"],
            "entity_2": ["in2"],
        },
    }

    if num_fc_layers is not None:
        config["combiner"]["num_fc_layers"] = num_fc_layers

    if fc_layers is not None:
        config["combiner"]["fc_layers"] = fc_layers

    with pytest.raises(ConfigValidationError) if not expect_success else contextlib.nullcontext():
        ModelConfig.from_dict(config)
