import logging
from typing import Dict, Tuple

from ludwig.constants import (
    CATEGORY,
    COMBINER,
    DECODER,
    DEFAULTS,
    ENCODER,
    FILL_WITH_CONST,
    INPUT_FEATURES,
    LOSS,
    OUTPUT_FEATURES,
    PREPROCESSING,
    TEXT,
    TRAINER,
    TYPE,
)
from ludwig.utils.defaults import merge_with_defaults
from tests.integration_tests.utils import category_feature, generate_data, run_experiment, text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


def _prepare_data(csv_filename: str) -> Tuple[Dict, str]:
    input_features = [
        text_feature(name="title", reduce_output="sum"),
        text_feature(name="summary"),
        category_feature(vocab_size=3),
        category_feature(vocab_size=3),
    ]

    output_features = [text_feature(name="article", embedding_size=3)]

    dataset = generate_data(input_features, output_features, csv_filename)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 1, "learning_rate": 0.001},
        DEFAULTS: {
            CATEGORY: {
                PREPROCESSING: {"missing_value_strategy": FILL_WITH_CONST, "fill_value": "<CUSTOM_TOK>"},
                ENCODER: {TYPE: "sparse"},
                DECODER: {"norm_params": None, "dropout": 0.1, "use_bias": True},
            },
            TEXT: {
                PREPROCESSING: {"most_common": 10, "padding_symbol": "<PADDING>"},
                ENCODER: {TYPE: "rnn"},
                DECODER: {TYPE: "generator", "num_fc_layers": 2, "dropout": 0.1},
                LOSS: {"confidence_penalty": 0.1},
            },
        },
    }

    return config, dataset


def test_run_experiment_with_global_default_parameters(csv_filename):
    config, dataset = _prepare_data(csv_filename)

    run_experiment(config=config, dataset=dataset)


def test_run_global_default_parameters_validate_config(csv_filename):
    config, _ = _prepare_data(csv_filename)

    updated_config = merge_with_defaults(config)

    assert DEFAULTS in updated_config

    assert CATEGORY not in updated_config[PREPROCESSING]
    assert TEXT not in updated_config[PREPROCESSING]

    for feature in updated_config[INPUT_FEATURES]:
        if feature[TYPE] == TEXT:
            assert feature[ENCODER] == "rnn"
        elif feature[TYPE] == CATEGORY:
            assert feature[ENCODER] == "sparse"

    assert updated_config[OUTPUT_FEATURES][0][DECODER] == "generator"
