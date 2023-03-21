import pytest

from ludwig.constants import DEFAULTS, ENCODER, INPUT_FEATURES, NAME, OUTPUT_FEATURES, SEQUENCE, TEXT, TIMESERIES, TYPE
from ludwig.error import ConfigValidationError
from ludwig.schema.model_config import ModelConfig
from tests.integration_tests.utils import (
    binary_feature,
    number_feature,
    sequence_feature,
    text_feature,
    timeseries_feature,
)


@pytest.mark.parametrize("feature_type", [SEQUENCE, TEXT, TIMESERIES])
def test_default_transformer_encoder(feature_type):
    """Tests that a transformer hyperparameter divisibility error is correctly recognized in feature defaults.

    Transformers require that `hidden_size % num_heads == 0`. 9 and 18 were selected as test values because they were
    the values from the original error.
    """
    config = {
        INPUT_FEATURES: [number_feature(), {TYPE: feature_type, NAME: f"test_{feature_type}"}],
        OUTPUT_FEATURES: [binary_feature()],
        DEFAULTS: {feature_type: {ENCODER: {TYPE: "transformer", "hidden_size": 9, "num_heads": 18}}},
    }

    with pytest.raises(ConfigValidationError):
        m = ModelConfig.from_dict(config)
        print(m)

    config[DEFAULTS][feature_type][ENCODER]["hidden_size"] = 18
    config[DEFAULTS][feature_type][ENCODER]["num_heads"] = 9

    ModelConfig.from_dict(config)


@pytest.mark.parametrize("feature_gen", [sequence_feature, text_feature, timeseries_feature])
def test_input_feature_transformer_encoder(feature_gen):
    """Tests that a transformer hyperparameter divisibility error is correctly recognized for a specific feature.

    Transformers require that `hidden_size % num_heads == 0`. 9 and 18 were selected as test values because they were
    the values from the original error.
    """
    config = {
        INPUT_FEATURES: [
            number_feature(),
            feature_gen(**{ENCODER: {TYPE: "transformer", "hidden_size": 9, "num_heads": 18}}),
        ],
        OUTPUT_FEATURES: [binary_feature()],
    }

    with pytest.raises(ConfigValidationError):
        ModelConfig.from_dict(config)

    config[INPUT_FEATURES][1][ENCODER]["hidden_size"] = 18
    config[INPUT_FEATURES][1][ENCODER]["num_heads"] = 9

    ModelConfig.from_dict(config)
