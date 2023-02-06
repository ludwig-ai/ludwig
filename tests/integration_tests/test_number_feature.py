import pandas as pd
import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import generate_data, number_feature


def test_number_feature_zscore_normalization_error():
    input_features = [number_feature(name="num_input", preprocessing={"normalization": "zscore"})]
    output_features = [number_feature(name="num_output")]

    df = pd.read_csv(generate_data(input_features, output_features))

    # Override input number feature to have a constant value
    df["num_input"] = 1

    config = {
        "input_features": input_features,
        "output_features": output_features,
    }

    model = LudwigModel(config, backend="local")

    with pytest.raises(RuntimeError):
        model.preprocess(dataset=df)


def test_number_feature_zscore_preprocessing_default():
    """Tests that the default value for the number feature preprocessing is 'zscore'"""
    input_features = [number_feature(name="num_input")]
    output_features = [number_feature(name="num_output")]

    input_features[0]["preprocessing"].pop("normalization")

    config = {
        "input_features": input_features,
        "output_features": output_features,
    }

    model = LudwigModel(config, backend="local")

    assert model.config["input_features"][0]["preprocessing"]["normalization"] == "zscore"
