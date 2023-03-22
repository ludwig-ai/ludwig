from ludwig.config_validation.validation import check_schema
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

    check_schema(config)

    # TODO(ksbrar): Circle back after discussing whether additional properties should be allowed long-term.
    # config["preprocessing"]["fake_parameter"] = True

    # with pytest.raises(Exception):
    #     ModelConfig(config)
