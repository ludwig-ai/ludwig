from typing import Union
from ludwig.constants import ACCURACY
from ludwig.schema.config.model_metadata import ModelMetadata


def check_membership(obj, attributes: Union[list, str], contains: bool = True):
    """ Helper function to determine if a metadata object or nested metadata
        object has or doesn't have a certain attribute"""
    if isinstance(attributes, str):
        attributes = list(attributes)

    for attr in attributes:
        # Checking if an object contains an attribute
        if contains and not hasattr(obj, attr):
            return False

        # Checking if an object does not have an attribute
        if not contains and hasattr(obj, attr):
            return False

    return True


def test_config_object():
    config = {
        "input_features": [
            {
                "name": "text_feature",
                "type": "text",
                "preprocessing": {
                    "missing_value_strategy": "drop_row",
                },
            },
            {
                "name": "image_feature_1",
                "type": "image",
                "encoder": {
                    "type": "resnet",
                    "num_channels": 4,
                    "dropout": 0.1,
                    "resnet_size": 100,
                },
            },
        ],
        "output_features": [
            {
                "name": "category_feature",
                "type": "category",
                "top_k": 3,
                "preprocessing": {
                    "missing_value_strategy": "bfill",
                },
                "decoder": {
                    "type": "classifier",
                    "num_classes": 10,
                    "use_bias": False,
                },
            },
        ],
        "trainer": {
            "optimizer": {
                "type": "adam",
            },
        },
    }

    model_metadata = ModelMetadata.from_dict(config)

    # Check input feature metadata contains proper field and doesn't contain anything it shouldn't
    assert len(model_metadata.input_features.to_list()) == 2
    check_membership(
        model_metadata.input_features.text_feature,
        ["name", "type", "column", "proc_column"]
    )
    check_membership(
        model_metadata.input_features.text_feature.preprocessing,
        "computed_fill_value"
    )
    check_membership(
        model_metadata.input_features.text_feature.preprocessing,
        "missing_value_strategy",
        contains=False
    )
    check_membership(
        model_metadata.input_features.image_feature_1.encoder,
        ["vocab", "vocab_size", "should_embed"]
    )
    check_membership(
        model_metadata.input_features.image_feature_1.encoder,
        ["num_channels", "dropout", "resnet_size"],
        contains=False
    )

    # Check output feature metadata contains proper field and doesn't contain anything it shouldn't
    assert len(model_metadata.output_features.to_list()) == 1
    assert model_metadata.output_features.category_feature.default_validation_metric == ACCURACY
    check_membership(
        model_metadata.output_features.category_feature,
        ["name", "type", "column", "proc_column", "default_validation_metric", "input_size", "num_classes"],
    )
    check_membership(
        model_metadata.output_features.category_feature,
        "top_k",
        contains=False
    )
    check_membership(
        model_metadata.output_features.category_feature.preprocessing,
        "computed_fill_value")
    check_membership(
        model_metadata.output_features.category_feature.preprocessing,
        "missing_value_strategy",
        contains=False
    )
    check_membership(
        model_metadata.output_features.category_feature.decoder,
        "vocab_size"
    )
    check_membership(
        model_metadata.output_features.category_feature.decoder,
        ["type", "num_classes", "use_bias"],
        contains=False
    )

    # Check trainer metadata contains proper field and doesn't contain anything it shouldn't
    check_membership(
        model_metadata.trainer.optimizer,
        "lr"
    )
    check_membership(
        model_metadata.trainer.optimizer,
        "type",
        contains=False
    )
