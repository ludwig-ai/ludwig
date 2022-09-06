import pytest
from ludwig.schema.config_object import Config


def test_config_object():
    config = {
        "input_features": [
            {
                "name": "text_feature",
                "type": "text",
                "preprocessing": {
                    "missing_value_strategy": "drop_row",
                },
                "encoder": {
                    "type": "rnn",
                    "bidirectional": True,
                    "representation": "dense",
                    "num_layers": 2,
                },
            },
            {
                "name": "image_feature_1",
                "type": "image",
                "preprocessing": {
                    "height": 7.5,
                    "width": 7.5,
                    "num_channels": 4,
                },
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
                    "missing_value_strategy": "backfill",
                },
                "decoder": {
                    "type": "classifier",
                    "num_classes": 10,
                    "use_bias": False,
                },
            },
        ],
        "combiner": {
            "type": "concat",
            "output_size": 512,
            "weights_initializer": "xavier_uniform",
            "dropout": 0.2,
        },
        "trainer": {
            "epochs": 50,
            "batch_size": "auto",
            "optimizer": {
                "type": "adam",
                "beta1": 0.8,
                "beta2": 0.999,
                "epsilon": 5e-09,
            }
        },
    }

    config_object = Config(config)
    assert config_object.input_features.text_feature.encoder.type == "rnn"
    assert config_object.input_features.text_feature.encoder.num_layers == 2

    assert config_object.output_features.category_feature.decoder.num_classes == 10
    assert config_object.output_features.category_feature.top_k == 3

    assert config_object.combiner.output_size == 512
    assert config_object.combiner.weights_initializer == "xavier_uniform"
    assert config_object.combiner.fc_layers is None

    assert config_object.trainer.epochs == 50
    assert config_object.trainer.batch_size == "auto"

    assert config_object.trainer.optimizer.type == "adam"
    assert config_object.trainer.optimizer.beta1 == 0.8
    assert config_object.trainer.optimizer.beta2 == 0.999
    assert config_object.trainer.optimizer.epsilon == 5e-09