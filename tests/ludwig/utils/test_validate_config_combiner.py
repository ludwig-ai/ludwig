import pytest
from jsonschema.exceptions import ValidationError

from ludwig.constants import TRAINER
from ludwig.schema import validate_config
from tests.integration_tests.utils import binary_feature, category_feature, number_feature


@pytest.mark.parametrize("eval_batch_size", [500000, None])
def test_config_tabnet(eval_batch_size):
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {
            "type": "tabnet",
            "size": 24,
            "output_size": 26,
            "sparsity": 0.000001,
            "bn_virtual_divider": 32,
            "bn_momentum": 0.6,
            "num_steps": 5,
            "relaxation_factor": 1.5,
            "use_keras_batch_norm": False,
            "bn_virtual_bs": 512,
        },
        TRAINER: {
            "batch_size": 16384,
            "eval_batch_size": eval_batch_size,
            "epochs": 1000,
            "early_stop": 20,
            "learning_rate": 0.02,
            "optimizer": {"type": "adam"},
            "decay": True,
            "decay_steps": 20000,
            "decay_rate": 0.9,
            "staircase": True,
            "regularization_lambda": 1,
            "regularization_type": "l2",
            "validation_field": "label",
        },
    }
    validate_config(config)


def test_config_bad_combiner():
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {
            "type": "tabnet",
        },
    }

    # config is valid at this point
    validate_config(config)

    # combiner without type
    del config["combiner"]["type"]
    with pytest.raises(ValidationError, match=r"^'type' is a required .*"):
        validate_config(config)

    # bad combiner type
    config["combiner"]["type"] = "fake"
    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)

    # bad combiner format (list instead of dict)
    config["combiner"] = [{"type": "tabnet"}]
    with pytest.raises(ValidationError, match=r"^\[\{'type': 'tabnet'\}\] is not of .*"):
        validate_config(config)

    # bad combiner parameter types
    config["combiner"] = {
        "type": "tabtransformer",
        "num_layers": 10,
        "dropout": False,
    }
    with pytest.raises(ValidationError, match=r"^False is not of type.*"):
        validate_config(config)

    # bad combiner parameter range
    config["combiner"] = {
        "type": "transformer",
        "dropout": -1,
    }
    with pytest.raises(ValidationError, match=r"less than the minimum.*"):
        validate_config(config)


def test_config_bad_combiner_types_enums():
    config = {
        "input_features": [
            category_feature(vocab_size=2, reduce_input="sum"),
            number_feature(),
        ],
        "output_features": [binary_feature(weight_regularization=None)],
        "combiner": {"type": "concat", "weights_initializer": "zeros"},
    }

    # config is valid at this point
    validate_config(config)

    # Test weights initializer:
    config["combiner"]["weights_initializer"] = {"test": "fail"}
    with pytest.raises(ValidationError, match=r"{'test': 'fail'} is not of*"):
        validate_config(config)
    config["combiner"]["weights_initializer"] = "fail"
    with pytest.raises(ValidationError, match=r"'fail' is not of*"):
        validate_config(config)
    config["combiner"]["weights_initializer"] = {}
    with pytest.raises(ValidationError, match=r"Failed validating 'type'"):
        validate_config(config)
    config["combiner"]["weights_initializer"] = {"type": "fail"}
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)
    config["combiner"]["weights_initializer"] = {"type": "normal", "stddev": 0}
    validate_config(config)

    # Test bias initializer:
    del config["combiner"]["weights_initializer"]
    config["combiner"]["bias_initializer"] = "kaiming_uniform"
    validate_config(config)
    config["combiner"]["bias_initializer"] = "fail"
    with pytest.raises(ValidationError, match=r"'fail' is not of*"):
        validate_config(config)
    config["combiner"]["bias_initializer"] = {}
    with pytest.raises(ValidationError, match=r"Failed validating 'type'"):
        validate_config(config)
    config["combiner"]["bias_initializer"] = {"type": "fail"}
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)
    config["combiner"]["bias_initializer"] = {"type": "zeros", "stddev": 0}
    validate_config(config)

    # Test norm:
    del config["combiner"]["bias_initializer"]
    config["combiner"]["norm"] = "batch"
    validate_config(config)
    config["combiner"]["norm"] = "fail"
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)

    # Test activation:
    del config["combiner"]["norm"]
    config["combiner"]["activation"] = "relu"
    validate_config(config)
    config["combiner"]["activation"] = 123
    with pytest.raises(ValidationError, match=r"123 is not of type*"):
        validate_config(config)

    # Test reduce_output:
    del config["combiner"]["activation"]
    config2 = {**config}
    config2["combiner"]["type"] = "tabtransformer"
    config2["combiner"]["reduce_output"] = "sum"
    validate_config(config)
    config2["combiner"]["reduce_output"] = "fail"
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config2)

    # Test reduce_output = None:
    config2["combiner"]["reduce_output"] = None
    validate_config(config2)
