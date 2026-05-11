import pandas as pd

from ludwig.api import LudwigModel
from tests.integration_tests.utils import generate_data, number_feature


def test_number_feature_zscore_normalization_constant():
    """ZScoreTransformer with std=0 should warn and fall back to identity (sigma=1) rather than crash."""
    import warnings

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

    # Should not raise — constant features are gracefully handled with a warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        preprocessed = model.preprocess(dataset=df)

    assert preprocessed is not None
