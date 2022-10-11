import copy

import pytest
from marshmallow import ValidationError

from ludwig.constants import (
    CATEGORY,
    DECODER,
    DEFAULTS,
    DEPENDENCIES,
    DROP_ROW,
    ENCODER,
    EXECUTOR,
    FILL_WITH_MODE,
    HYPEROPT,
    INPUT_FEATURES,
    MAX_POSSIBLE_BATCH_SIZE,
    MISSING_VALUE_STRATEGY,
    MODEL_ECD,
    MODEL_GBM,
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
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.utils.backward_compatibility import upgrade_to_latest_version
from ludwig.utils.defaults import merge_with_defaults
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

    merged_config = merge_with_defaults(config)

    # When a scheulder is provided, early stopping in the rendered config needs to be disabled to allow the
    # hyperopt scheduler to manage trial lifecycle.
    expected = -1 if use_hyperopt_scheduler else ECDTrainerConfig().early_stop
    assert merged_config[TRAINER]["early_stop"] == expected


def test_missing_outputs_drop_rows():
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature()],
        DEFAULTS: {CATEGORY: {PREPROCESSING: {MISSING_VALUE_STRATEGY: FILL_WITH_MODE}}},
    }

    merged_config = merge_with_defaults(config)

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
        OUTPUT_FEATURES: [category_feature()],
    }

    merged_config = merge_with_defaults(config)

    assert merged_config[MODEL_TYPE] == MODEL_ECD


@pytest.mark.parametrize(
    "model_trainer_type",
    [
        (MODEL_ECD, "trainer"),
        (MODEL_GBM, "lightgbm_trainer"),
    ],
)
def test_default_trainer_type(model_trainer_type):
    model_type, expected_trainer_type = model_trainer_type
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature()],
        MODEL_TYPE: model_type,
    }

    merged_config = merge_with_defaults(config)

    assert merged_config[TRAINER][TYPE] == expected_trainer_type


def test_overwrite_trainer_type():
    expected_trainer_type = "ray_legacy_trainer"
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature()],
        MODEL_TYPE: MODEL_ECD,
        "trainer": {"type": expected_trainer_type},
    }

    merged_config = merge_with_defaults(config)

    assert merged_config[TRAINER][TYPE] == expected_trainer_type


@pytest.mark.parametrize(
    "model_type",
    [MODEL_ECD, MODEL_GBM],
)
def test_invalid_trainer_type(model_type):
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature()],
        MODEL_TYPE: model_type,
        "trainer": {"type": "invalid_trainer"},
    }

    with pytest.raises(ValidationError):
        merge_with_defaults(config)


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
            {
                "type": "numerical",
                "name": "number_input_feature",
            },
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
                "training.learning_rate": {},
                "training.early_stop": {},
                "number_input_feature.num_fc_layers": {},
                "number_output_feature.embedding_size": {},
                "number_output_feature.dropout": 0.2,
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

    # expected configuration content with default values after upgrading legacy configuration components
    expected_upgraded_format = {
        "ludwig_version": LUDWIG_VERSION,
        "input_features": [
            {
                "type": "number",
                "name": "number_input_feature",
                "column": "number_input_feature",
                "proc_column": "number_input_feature_mZFLky",
                "tied": None,
                "encoder": {"type": "passthrough"},
                "preprocessing": {
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": 0.0,
                    "computed_fill_value": 0.0,
                    "normalization": None,
                },
            },
            {
                "type": "image",
                "name": "image_input_feature",
                "encoder": {
                    "type": "stacked_cnn",
                    "height": None,
                    "width": None,
                    "conv_layers": [
                        {"num_filters": 32, "pool_size": 2, "pool_stride": 2, "use_bias": False},
                        {"num_filters": 64, "pool_size": 2, "pool_stride": 2},
                    ],
                    "num_conv_layers": None,
                    "num_channels": None,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": "valid",
                    "dilation": 1,
                    "groups": 1,
                    "conv_use_bias": True,
                    "padding_mode": "zeros",
                    "conv_norm": None,
                    "conv_norm_params": None,
                    "conv_activation": "relu",
                    "conv_dropout": 0.0,
                    "pool_function": "max",
                    "pool_kernel_size": 2,
                    "pool_stride": None,
                    "pool_padding": 0,
                    "pool_dilation": 1,
                    "fc_layers": None,
                    "num_fc_layers": 1,
                    "output_size": 128,
                    "fc_use_bias": True,
                    "fc_weights_initializer": "xavier_uniform",
                    "fc_bias_initializer": "zeros",
                    "fc_norm": None,
                    "fc_norm_params": None,
                    "fc_activation": "relu",
                    "fc_dropout": 0.0,
                },
                "column": "image_input_feature",
                "proc_column": "image_input_feature_mZFLky",
                "tied": None,
                "preprocessing": {
                    "missing_value_strategy": "bfill",
                    "fill_value": None,
                    "computed_fill_value": None,
                    "height": None,
                    "width": None,
                    "num_channels": None,
                    "resize_method": "interpolate",
                    "infer_image_num_channels": True,
                    "infer_image_dimensions": True,
                    "infer_image_max_height": 256,
                    "infer_image_max_width": 256,
                    "infer_image_sample_size": 100,
                    "scaling": "pixel_normalization",
                    "in_memory": True,
                    "num_processes": 1,
                },
            },
        ],
        "output_features": [
            {
                "type": "number",
                "name": "number_output_feature",
                "column": "number_output_feature",
                "proc_column": "number_output_feature_mZFLky",
                "loss": {"type": "mean_squared_error", "weight": 1},
                "decoder": {
                    "type": "regressor",
                    "fc_layers": None,
                    "num_fc_layers": 0,
                    "fc_output_size": 256,
                    "fc_use_bias": True,
                    "fc_weights_initializer": "xavier_uniform",
                    "fc_bias_initializer": "zeros",
                    "fc_norm": None,
                    "fc_norm_params": None,
                    "fc_activation": "relu",
                    "fc_dropout": 0.0,
                    "input_size": None,
                    "use_bias": True,
                    "weights_initializer": "xavier_uniform",
                    "bias_initializer": "zeros",
                },
                "clip": None,
                "dependencies": [],
                "reduce_input": "sum",
                "reduce_dependencies": "sum",
                "preprocessing": {
                    "missing_value_strategy": "drop_row",
                    "fill_value": 0,
                    "computed_fill_value": 0,
                    "normalization": None,
                },
                "input_size": None,
                "num_classes": None,
            }
        ],
        "hyperopt": {
            "parameters": {
                "number_input_feature.num_fc_layers": {},
                "number_output_feature.embedding_size": {},
                "number_output_feature.dropout": 0.2,
                "trainer.learning_rate": {},
                "trainer.early_stop": {},
            },
            "executor": {"type": "ray", "num_samples": 99, "scheduler": {}},
            "search_alg": {"type": "variant_generator"},
        },
        "trainer": {
            "type": "trainer",
            "learning_rate": 0.001,
            "validation_metric": "loss",
            "validation_field": "combined",
            "eval_batch_size": None,
            "early_stop": 5,
            "evaluate_training_set": True,
            "optimizer": {"type": "adadelta", "lr": 1.0, "rho": 0.9, "eps": 1e-06, "weight_decay": 0.0},
            "epochs": 100,
            "train_steps": None,
            "regularization_lambda": 0.0,
            "regularization_type": "l2",
            "should_shuffle": True,
            "batch_size": 128,
            "max_batch_size": MAX_POSSIBLE_BATCH_SIZE,
            "steps_per_checkpoint": 0,
            "checkpoints_per_epoch": 0,
            "reduce_learning_rate_on_plateau": 0.0,
            "reduce_learning_rate_on_plateau_patience": 5,
            "reduce_learning_rate_on_plateau_rate": 0.5,
            "reduce_learning_rate_eval_metric": "loss",
            "reduce_learning_rate_eval_split": "training",
            "increase_batch_size_on_plateau": 0,
            "increase_batch_size_on_plateau_patience": 5,
            "increase_batch_size_on_plateau_rate": 2.0,
            "increase_batch_size_eval_metric": "loss",
            "increase_batch_size_eval_split": "training",
            "decay": False,
            "decay_steps": 10000,
            "decay_rate": 0.96,
            "staircase": False,
            "gradient_clipping": {"clipglobalnorm": 0.5, "clipnorm": None, "clipvalue": None},
            "learning_rate_warmup_epochs": 1.0,
            "learning_rate_scaling": "linear",
        },
        "defaults": {
            "text": {
                "preprocessing": {
                    "pretrained_model_name_or_path": None,
                    "tokenizer": "space_punct",
                    "vocab_file": None,
                    "max_sequence_length": 256,
                    "most_common": 20000,
                    "padding_symbol": "<PAD>",
                    "unknown_symbol": "<UNK>",
                    "padding": "right",
                    "lowercase": True,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                    "computed_fill_value": "<UNK>",
                }
            },
            "category": {
                "preprocessing": {
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                    "computed_fill_value": "<UNK>",
                    "lowercase": False,
                    "most_common": 10000,
                }
            },
            "set": {
                "preprocessing": {
                    "tokenizer": "space",
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                    "computed_fill_value": "<UNK>",
                    "lowercase": False,
                    "most_common": 10000,
                }
            },
            "bag": {
                "preprocessing": {
                    "tokenizer": "space",
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                    "computed_fill_value": "<UNK>",
                    "lowercase": False,
                    "most_common": 10000,
                }
            },
            "binary": {
                "preprocessing": {
                    "missing_value_strategy": "fill_with_false",
                    "fill_value": None,
                    "computed_fill_value": None,
                    "fallback_true_label": None,
                }
            },
            "number": {
                "preprocessing": {
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": 0.0,
                    "computed_fill_value": 0.0,
                    "normalization": None,
                }
            },
            "sequence": {
                "preprocessing": {
                    "tokenizer": "space",
                    "vocab_file": None,
                    "max_sequence_length": 256,
                    "most_common": 20000,
                    "padding_symbol": "<PAD>",
                    "unknown_symbol": "<UNK>",
                    "padding": "right",
                    "lowercase": False,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                    "computed_fill_value": "<UNK>",
                }
            },
            "timeseries": {
                "preprocessing": {
                    "tokenizer": "space",
                    "timeseries_length_limit": 256,
                    "padding_value": 0.0,
                    "padding": "right",
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "",
                    "computed_fill_value": "",
                }
            },
            "image": {
                "preprocessing": {
                    "missing_value_strategy": "bfill",
                    "fill_value": None,
                    "computed_fill_value": None,
                    "height": None,
                    "width": None,
                    "num_channels": None,
                    "resize_method": "interpolate",
                    "infer_image_num_channels": True,
                    "infer_image_dimensions": True,
                    "infer_image_max_height": 256,
                    "infer_image_max_width": 256,
                    "infer_image_sample_size": 100,
                    "scaling": "pixel_normalization",
                    "in_memory": True,
                    "num_processes": 1,
                }
            },
            "audio": {
                "preprocessing": {
                    "audio_file_length_limit_in_s": 7.5,
                    "missing_value_strategy": "bfill",
                    "fill_value": None,
                    "computed_fill_value": None,
                    "in_memory": True,
                    "padding_value": 0.0,
                    "norm": None,
                    "type": "fbank",
                    "window_length_in_s": 0.04,
                    "window_shift_in_s": 0.02,
                    "num_fft_points": None,
                    "window_type": "hamming",
                    "num_filter_bands": 80,
                }
            },
            "h3": {
                "preprocessing": {
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": 576495936675512319,
                    "computed_fill_value": 576495936675512319,
                }
            },
            "date": {
                "preprocessing": {
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "",
                    "computed_fill_value": "",
                    "datetime_format": None,
                }
            },
            "vector": {
                "preprocessing": {
                    "vector_size": None,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "",
                    "computed_fill_value": "",
                }
            },
        },
        "preprocessing": {
            "split": {"type": "random", "probabilities": [0.7, 0.1, 0.2]},
            "undersample_majority": None,
            "oversample_minority": None,
            "sample_ratio": 1.0,
        },
        "model_type": "ecd",
        "combiner": {
            "type": "concat",
            "fc_layers": None,
            "num_fc_layers": 0,
            "output_size": 256,
            "use_bias": True,
            "weights_initializer": "xavier_uniform",
            "bias_initializer": "zeros",
            "norm": None,
            "norm_params": None,
            "activation": "relu",
            "dropout": 0.0,
            "flatten_inputs": False,
            "residual": False,
        },
    }

    updated_config = upgrade_to_latest_version(legacy_config_format)
    merged_config = merge_with_defaults(updated_config)

    assert merged_config == expected_upgraded_format
