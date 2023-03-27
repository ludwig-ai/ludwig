import copy

import pytest

from ludwig.constants import (
    CATEGORY,
    COMBINER,
    DECODER,
    DEFAULTS,
    DEPENDENCIES,
    DROP_ROW,
    EARLY_STOP,
    ENCODER,
    EXECUTOR,
    FILL_WITH_MODE,
    HYPEROPT,
    INPUT_FEATURES,
    LOSS,
    MISSING_VALUE_STRATEGY,
    MODEL_ECD,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PREPROCESSING,
    REDUCE_DEPENDENCIES,
    REDUCE_INPUT,
    SCHEDULER,
    SUM,
    TIED,
    TOP_K,
    TRAINER,
    TYPE,
)
from ludwig.schema.model_config import ModelConfig
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.misc_utils import merge_dict, set_default_values
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
        "utterance.encoder.norm": {"space": "grid_search", "values": ["layer", "batch"]},
        "utterance.encoder.dropout": {"space": "choice", "categories": [0.0001, 0.001, 0.01]},
        "utterance.encoder.fc_layers": {
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
        text_feature(name="utterance"),
    ]
    all_output_features = [
        category_feature(output_feature=True),
        sequence_feature(output_feature=True),
        vector_feature(),
    ]

    # validate config with all features
    config = {
        INPUT_FEATURES: all_input_features,
        OUTPUT_FEATURES: all_output_features,
        HYPEROPT: HYPEROPT_CONFIG,
    }
    config = copy.deepcopy(config)

    if use_train:
        config[TRAINER] = {"batch_size": 42}

    if use_hyperopt_scheduler:
        # hyperopt scheduler cannot be used with early stopping
        config[HYPEROPT][EXECUTOR][SCHEDULER] = SCHEDULER_DICT

    merged_config = ModelConfig.from_dict(config).to_dict()

    # When a scheulder is provided, early stopping in the rendered config needs to be disabled to allow the
    # hyperopt scheduler to manage trial lifecycle.
    expected = -1 if use_hyperopt_scheduler else ECDTrainerConfig().early_stop
    assert merged_config[TRAINER]["early_stop"] == expected


def test_missing_outputs_drop_rows():
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature(output_feature=True)],
        DEFAULTS: {CATEGORY: {PREPROCESSING: {MISSING_VALUE_STRATEGY: FILL_WITH_MODE}}},
    }

    merged_config = ModelConfig.from_dict(config).to_dict()

    global_preprocessing = merged_config[DEFAULTS]
    input_feature_config = merged_config[INPUT_FEATURES][0]
    output_feature_config = merged_config[OUTPUT_FEATURES][0]

    assert output_feature_config[PREPROCESSING][MISSING_VALUE_STRATEGY] == DROP_ROW

    assert global_preprocessing[input_feature_config[TYPE]][PREPROCESSING][MISSING_VALUE_STRATEGY] == FILL_WITH_MODE
    feature_preprocessing = merge_dict(
        global_preprocessing[output_feature_config[TYPE]][PREPROCESSING], output_feature_config[PREPROCESSING]
    )
    assert feature_preprocessing[MISSING_VALUE_STRATEGY] == DROP_ROW


def test_default_model_type():
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature(output_feature=True)],
    }

    merged_config = ModelConfig.from_dict(config).to_dict()

    assert merged_config[MODEL_TYPE] == MODEL_ECD


def test_set_default_values():
    config = {
        INPUT_FEATURES: [number_feature(encoder={"max_sequence_length": 10})],
        OUTPUT_FEATURES: [category_feature(decoder={})],
    }

    assert TIED not in config[INPUT_FEATURES][0]
    assert TOP_K not in config[OUTPUT_FEATURES][0]
    assert DEPENDENCIES not in config[OUTPUT_FEATURES][0]
    assert REDUCE_INPUT not in config[OUTPUT_FEATURES][0]
    assert REDUCE_DEPENDENCIES not in config[OUTPUT_FEATURES][0]

    set_default_values(config[INPUT_FEATURES][0], {ENCODER: {TYPE: "passthrough"}, TIED: None})

    set_default_values(
        config[OUTPUT_FEATURES][0],
        {
            DECODER: {
                TYPE: "classifier",
            },
            TOP_K: 3,
            DEPENDENCIES: [],
            REDUCE_INPUT: SUM,
            REDUCE_DEPENDENCIES: SUM,
        },
    )

    assert config[INPUT_FEATURES][0][ENCODER][TYPE] == "passthrough"
    assert config[INPUT_FEATURES][0][TIED] is None
    assert config[OUTPUT_FEATURES][0][DECODER][TYPE] == "classifier"
    assert config[OUTPUT_FEATURES][0][TOP_K] == 3
    assert config[OUTPUT_FEATURES][0][DEPENDENCIES] == []
    assert config[OUTPUT_FEATURES][0][REDUCE_INPUT] == SUM
    assert config[OUTPUT_FEATURES][0][REDUCE_DEPENDENCIES] == SUM


def test_merge_with_defaults():
    # configuration with legacy parameters
    legacy_config_format = {
        "ludwig_version": "0.4",
        INPUT_FEATURES: [
            {"type": "numerical", "name": "number_input_feature", "encoder": {"type": "dense"}},
            {
                "type": "image",
                "name": "image_input_feature",
                "encoder": "stacked_cnn",
                "conv_bias": True,
                "conv_layers": [
                    {"num_filters": 32, "pool_size": 2, "pool_stride": 2, "bias": False},
                    {
                        "num_filters": 64,
                        "pool_size": 2,
                        "pool_stride": 2,
                    },
                ],
            },
        ],
        OUTPUT_FEATURES: [
            {
                "type": "numerical",
                "name": "number_output_feature",
            },
        ],
        "training": {"eval_batch_size": 0, "optimizer": {"type": "adadelta"}},
        HYPEROPT: {
            "parameters": {
                "training.learning_rate": {"space": "choice", "categories": [0.0001, 0.001, 0.01]},
                "training.early_stop": {"space": "choice", "categories": [5, 10, 15]},
                "number_input_feature.encoder.num_layers": {"space": "choice", "categories": [2, 3, 4]},
                "number_output_feature.decoder.fc_output_size": {"space": "choice", "categories": [128, 256, 512]},
                "number_output_feature.decoder.fc_dropout": {"space": "uniform", "lower": 0, "upper": 1},
            },
            "executor": {
                "type": "serial",
                "search_alg": {TYPE: "variant_generator"},
            },
            "sampler": {
                "num_samples": 99,
                "scheduler": {},
            },
        },
    }

    updated_config = upgrade_config_dict_to_latest_version(legacy_config_format)
    merged_config = ModelConfig.from_dict(updated_config).to_dict()

    assert len(merged_config[DEFAULTS]) == 13
    assert ENCODER in merged_config[DEFAULTS][CATEGORY]
    assert PREPROCESSING in merged_config[DEFAULTS][CATEGORY]
    assert DECODER in merged_config[DEFAULTS][CATEGORY]
    assert LOSS in merged_config[DEFAULTS][CATEGORY]
    assert COMBINER in merged_config
    assert merged_config[TRAINER][EARLY_STOP] == 5
    assert SCHEDULER in merged_config[HYPEROPT][EXECUTOR]
    assert merged_config[HYPEROPT][EXECUTOR][SCHEDULER]["type"] == "fifo"
    assert TYPE in merged_config[INPUT_FEATURES][1][ENCODER]
    assert TYPE in merged_config[OUTPUT_FEATURES][0][DECODER]
