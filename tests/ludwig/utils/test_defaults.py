import copy

import pytest

from ludwig.constants import (
    CATEGORY,
    DROP_ROW,
    EVAL_BATCH_SIZE,
    FILL_WITH_MODE,
    HYPEROPT,
    NUMBER,
    PREPROCESSING,
    TRAINER,
    TYPE,
)
from ludwig.data.preprocessing import merge_preprocessing
from ludwig.utils.defaults import default_training_params, merge_with_defaults
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    number_feature,
    sequence_feature,
    text_feature,
    vector_feature,
)

HYPEROPT_CONFIG = {
    "parameters": {
        "trainer.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.num_fc_layers": {"space": "randint", "lower": 2, "upper": 6},
        "utterance.cell_type": {"space": "grid_search", "values": ["rnn", "gru"]},
        "utterance.bidirectional": {"space": "choice", "categories": [True, False]},
        "utterance.fc_layers": {
            "space": "choice",
            "categories": [
                [{"output_size": 512}, {"output_size": 256}],
                [{"output_size": 512}],
                [{"output_size": 256}],
            ],
        },
    },
    "sampler": {"type": "ray"},
    "executor": {"type": "ray"},
    "goal": "minimize",
}

SCHEDULER = {"type": "async_hyperband", "time_attr": "time_total_s"}

default_early_stop = default_training_params["early_stop"]


@pytest.mark.parametrize(
    "use_train,use_hyperopt_scheduler",
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
)
def test_merge_with_defaults_early_stop(use_train, use_hyperopt_scheduler):
    all_input_features = [
        binary_feature(),
        category_feature(),
        number_feature(),
        text_feature(),
    ]
    all_output_features = [
        category_feature(),
        sequence_feature(),
        vector_feature(),
    ]

    # validate config with all features
    config = {
        "input_features": all_input_features,
        "output_features": all_output_features,
        HYPEROPT: HYPEROPT_CONFIG,
    }
    config = copy.deepcopy(config)

    if use_train:
        config[TRAINER] = {"batch_size": "42"}

    if use_hyperopt_scheduler:
        # hyperopt scheduler cannot be used with early stopping
        config[HYPEROPT]["sampler"]["scheduler"] = SCHEDULER

    merged_config = merge_with_defaults(config)

    expected = -1 if use_hyperopt_scheduler else default_early_stop
    assert merged_config[TRAINER]["early_stop"] == expected


def test_missing_outputs_drop_rows():
    config = {
        "input_features": [
            category_feature(),
        ],
        "output_features": [
            category_feature(),
        ],
        PREPROCESSING: {CATEGORY: {"missing_value_strategy": FILL_WITH_MODE}},
    }

    merged_config = merge_with_defaults(config)
    feature_config = merged_config["output_features"][0]
    assert feature_config[PREPROCESSING]["missing_value_strategy"] == DROP_ROW

    global_preprocessing = merged_config[PREPROCESSING]
    feature_preprocessing = merge_preprocessing(feature_config, global_preprocessing)
    assert feature_preprocessing["missing_value_strategy"] == DROP_ROW

    feature_preprocessing = merge_preprocessing(merged_config["input_features"][0], global_preprocessing)
    assert feature_preprocessing["missing_value_strategy"] == FILL_WITH_MODE


def test_deprecated_field_aliases():
    config = {
        "input_features": [{"name": "num_in", "type": "number"}],
        "output_features": [{"name": "num_out", "type": "number"}],
        "training": {
            "epochs": 2,
            "eval_batch_size": 0,
        },
        "hyperopt": {
            "parameters": {
                "training.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
            },
            "goal": "minimize",
        },
    }

    merged_config = merge_with_defaults(config)

    assert merged_config["input_features"][0][TYPE] == NUMBER
    assert merged_config["output_features"][0][TYPE] == NUMBER

    assert "training" not in merged_config
    assert merged_config[TRAINER]["epochs"] == 2
    assert merged_config[TRAINER][EVAL_BATCH_SIZE] is None

    hparams = merged_config[HYPEROPT]["parameters"]
    assert "training.learning_rate" not in hparams
    assert "trainer.learning_rate" in hparams
