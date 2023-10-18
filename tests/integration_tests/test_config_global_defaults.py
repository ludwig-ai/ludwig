import logging
from typing import Dict, Tuple

from ludwig.constants import (
    BATCH_SIZE,
    CATEGORY,
    COMBINER,
    DECODER,
    DEFAULTS,
    ENCODER,
    EPOCHS,
    FILL_WITH_CONST,
    INPUT_FEATURES,
    LOSS,
    OUTPUT_FEATURES,
    PREPROCESSING,
    TEXT,
    TRAINER,
    TYPE,
)
from ludwig.schema.model_config import ModelConfig
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

    output_features = [text_feature(name="article", embedding_size=3, output_feature=True)]

    dataset = generate_data(input_features, output_features, csv_filename)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {EPOCHS: 1, "learning_rate": 0.001, BATCH_SIZE: 128},
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


def test_global_defaults_with_encoder_dependencies():
    input_features = [text_feature(name="title", reduce_output="sum")]
    output_features = [category_feature(name="article", embedding_size=3, output_feature=True)]
    del input_features[0][ENCODER]

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        DEFAULTS: {
            TEXT: {
                ENCODER: {TYPE: "bert"},
            }
        },
    }

    # Config should populate with the additional required fields for bert
    updated_config = ModelConfig.from_dict(config).to_dict()

    assert updated_config[INPUT_FEATURES][0][ENCODER][TYPE] == "bert"
    assert updated_config[INPUT_FEATURES][0][ENCODER]["pretrained_model_name_or_path"] == "bert-base-uncased"
