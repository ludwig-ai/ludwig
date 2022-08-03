import logging
from typing import Dict, Tuple

from ludwig.constants import (
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
from ludwig.features.feature_registries import input_type_registry
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
        TRAINER: {EPOCHS: 1, "learning_rate": 0.001},
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


def test_global_default_parameters_merge_with_defaults(csv_filename):
    config, _ = _prepare_data(csv_filename)

    updated_config = merge_with_defaults(config)

    assert DEFAULTS in updated_config

    # Make sure no type specific parameters are in preprocessing
    input_feature_types = set(input_type_registry)
    for parameter in updated_config[PREPROCESSING]:
        assert parameter not in input_feature_types

    # All feature-specific preprocessing parameters should be in defaults
    defaults_with_preprocessing = [
        feature for feature in updated_config[DEFAULTS] if PREPROCESSING in updated_config[DEFAULTS][feature]
    ]
    assert len(defaults_with_preprocessing) == len(input_feature_types)

    # Feature encoders and decoders should update
    for feature in updated_config[INPUT_FEATURES]:
        assert feature[ENCODER] == updated_config[DEFAULTS][feature[TYPE]][ENCODER][TYPE]

    output_feature = updated_config[OUTPUT_FEATURES][0]
    assert output_feature[DECODER] == updated_config[DEFAULTS][output_feature[TYPE]][DECODER][TYPE]


def test_global_defaults_with_encoder_dependencies(csv_filename):
    input_features = [text_feature(name="title", reduce_output="sum")]
    output_features = [category_feature(name="article", embedding_size=3)]

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
    updated_config = merge_with_defaults(config)

    assert updated_config[INPUT_FEATURES][0][ENCODER] == "bert"
    assert updated_config[INPUT_FEATURES][0]["pretrained_model_name_or_path"] == "bert-base-uncased"
