import copy

import pytest

from ludwig.constants import (
    CATEGORY,
    DROP_ROW,
    EVAL_BATCH_SIZE,
    EXECUTOR,
    FILL_WITH_MODE,
    HYPEROPT,
    NUMBER,
    PARAMETERS,
    PREPROCESSING,
    RAY,
    SAMPLER,
    SCHEDULER,
    SEARCH_ALG,
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
    "search_alg": {"type": "hyperopt"},
    "executor": {"type": "ray"},
    "goal": "minimize",
}

SCHEDULER_DICT = {"type": "async_hyperband", "time_attr": "time_total_s"}

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
        config[HYPEROPT]["executor"][SCHEDULER] = SCHEDULER_DICT

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
            "sampler": {"type": "grid", "num_samples": 2, "scheduler": {"type": "fifo"}},
            "executor": {
                "type": "grid",
                "search_alg": "bohb",
            },
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

    assert "sampler" not in merged_config[HYPEROPT]

    assert merged_config[HYPEROPT]["executor"]["type"] == "ray"
    assert "num_samples" in merged_config[HYPEROPT]["executor"]
    assert "scheduler" in merged_config[HYPEROPT]["executor"]


def test_merge_with_defaults():
    legacy_config_format = {
        "input_features": [
            {"type": "numerical", "name": "in_feat", },
        ],
        "output_features": [
            {"type": "numerical", "name": "out_feat", },
        ],
        "training": {"eval_batch_size": 0},
        "hyperopt": {
            "parameters": {
                "training.learning_rate": {},
                "training.early_stop": {},
                "in_feat.num_fc_layers": {},
                "out_feat.embedding_size": {},
                "out_feat.dropout": 0.2,
            },
            "executor": {"type": "serial", "search_alg": {TYPE: "variant_generator"}, },
            "sampler": {"num_samples": 99, "scheduler": {}, },
        }
    }

    updated_config = merge_with_defaults(legacy_config_format)

    # check for updated trainer section
    assert TRAINER in updated_config and "training" not in updated_config
    assert updated_config[TRAINER]["eval_batch_size"] is None \
           and updated_config[TRAINER]["eval_batch_size"] != 0

    # check for updated number type for input and output features
    assert NUMBER == updated_config["input_features"][0][TYPE] \
           and "numerical" != updated_config["input_features"][0][TYPE]
    assert NUMBER == updated_config["output_features"][0][TYPE] \
           and "numerical" != updated_config["output_features"][0][TYPE]

    # check for upgraded hyperparameters
    assert "trainer.learning_rate" in updated_config[HYPEROPT][PARAMETERS] \
           and "training.learning_rate" not in updated_config[HYPEROPT][PARAMETERS]
    assert "trainer.early_stop" in updated_config[HYPEROPT][PARAMETERS] \
           and "training.early_stop" not in updated_config[HYPEROPT][PARAMETERS]

    # make sure other parameters not changed or missing
    assert "in_feat.num_fc_layers" in updated_config[HYPEROPT][PARAMETERS]
    assert "out_feat.embedding_size" in updated_config[HYPEROPT][PARAMETERS]
    assert "out_feat.dropout" in updated_config[HYPEROPT][PARAMETERS]

    # check hyperopt executor updates
    assert updated_config[HYPEROPT][EXECUTOR][TYPE] == RAY and updated_config[HYPEROPT][EXECUTOR][TYPE] != "serial"

    # check for search_alg
    assert SEARCH_ALG in updated_config[HYPEROPT] and SEARCH_ALG not in updated_config[HYPEROPT][EXECUTOR]
    assert "variant_generator" == updated_config[HYPEROPT][SEARCH_ALG][TYPE]

    # ensure sampler section no longer exists
    assert SAMPLER not in updated_config[HYPEROPT]

    # check for specified sampler parameters migrated to new location
    assert "num_samples" in updated_config[HYPEROPT][EXECUTOR] \
           and updated_config[HYPEROPT][EXECUTOR]["num_samples"] == 99
    assert SCHEDULER in updated_config[HYPEROPT][EXECUTOR]
