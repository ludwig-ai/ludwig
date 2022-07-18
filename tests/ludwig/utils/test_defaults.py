import copy

import pytest
from marshmallow import ValidationError

from ludwig.constants import (
    CATEGORY,
    DEFAULTS,
    DROP_ROW,
    EVAL_BATCH_SIZE,
    EXECUTOR,
    FILL_WITH_MODE,
    HYPEROPT,
    INPUT_FEATURES,
    MISSING_VALUE_STRATEGY,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_TYPE,
    NUMBER,
    OUTPUT_FEATURES,
    PREPROCESSING,
    SCHEDULER,
    SPLIT,
    TRAINER,
    TYPE,
)
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.utils.defaults import merge_with_defaults
from ludwig.utils.misc_utils import merge_dict
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

    expected = -1 if use_hyperopt_scheduler else ECDTrainerConfig().early_stop
    assert merged_config[TRAINER]["early_stop"] == expected


def test_missing_outputs_drop_rows():
    config = {
        INPUT_FEATURES: [category_feature()],
        OUTPUT_FEATURES: [category_feature()],
        PREPROCESSING: {CATEGORY: {MISSING_VALUE_STRATEGY: FILL_WITH_MODE}},
    }

    merged_config = merge_with_defaults(config)

    global_preprocessing = merged_config[DEFAULTS]
    input_feature_config = merged_config[INPUT_FEATURES][0]
    output_feature_config = merged_config[OUTPUT_FEATURES][0]

    assert output_feature_config[PREPROCESSING][MISSING_VALUE_STRATEGY] == DROP_ROW

    feature_preprocessing = merge_dict(
        global_preprocessing[output_feature_config[TYPE]][PREPROCESSING], output_feature_config[PREPROCESSING]
    )
    assert feature_preprocessing[MISSING_VALUE_STRATEGY] == DROP_ROW

    feature_preprocessing = global_preprocessing[input_feature_config[TYPE]][PREPROCESSING]
    assert feature_preprocessing[MISSING_VALUE_STRATEGY] == FILL_WITH_MODE


def test_deprecated_field_aliases():
    config = {
        INPUT_FEATURES: [{"name": "num_in", "type": "numerical"}],
        OUTPUT_FEATURES: [{"name": "num_out", "type": "numerical"}],
        "training": {
            "epochs": 2,
            "eval_batch_size": 0,
        },
        HYPEROPT: {
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


@pytest.mark.parametrize("force_split", [None, False, True])
@pytest.mark.parametrize("stratify", [None, "cat_in"])
def test_deprecated_split_aliases(stratify, force_split):
    split_probabilities = [0.6, 0.2, 0.2]
    config = {
        INPUT_FEATURES: [{"name": "num_in", "type": "number"}, {"name": "cat_in", "type": "category"}],
        OUTPUT_FEATURES: [{"name": "num_out", "type": "number"}],
        PREPROCESSING: {
            "force_split": force_split,
            "split_probabilities": split_probabilities,
            "stratify": stratify,
        },
    }

    merged_config = merge_with_defaults(config)

    assert "force_split" not in merged_config[PREPROCESSING]
    assert "split_probabilities" not in merged_config[PREPROCESSING]
    assert "stratify" not in merged_config[PREPROCESSING]

    assert SPLIT in merged_config[PREPROCESSING]
    split = merged_config[PREPROCESSING][SPLIT]

    assert split["probabilities"] == split_probabilities
    if stratify is None:
        if force_split:
            assert split.get(TYPE) == "random"
    else:
        assert split.get(TYPE) == "stratify"
        assert split.get("column") == stratify


def test_merge_with_defaults():
    # configuration with legacy parameters
    legacy_config_format = {
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
        MODEL_TYPE: "ecd",
        INPUT_FEATURES: [
            {
                "type": "number",
                "name": "number_input_feature",
                "column": "number_input_feature",
                "proc_column": "number_input_feature_mZFLky",
                "tied": None,
            },
            {
                "type": "image",
                "name": "image_input_feature",
                "encoder": "stacked_cnn",
                "conv_layers": [
                    {"num_filters": 32, "pool_size": 2, "pool_stride": 2, "use_bias": False},
                    {"num_filters": 64, "pool_size": 2, "pool_stride": 2},
                ],
                "conv_use_bias": True,
                "column": "image_input_feature",
                "proc_column": "image_input_feature_mZFLky",
                "tied": None,
                "preprocessing": {},
            },
        ],
        OUTPUT_FEATURES: [
            {
                "type": "number",
                "name": "number_output_feature",
                "column": "number_output_feature",
                "proc_column": "number_output_feature_mZFLky",
                "loss": {"type": "mean_squared_error", "weight": 1},
                "clip": None,
                "dependencies": [],
                "reduce_input": "sum",
                "reduce_dependencies": "sum",
                "preprocessing": {"missing_value_strategy": "drop_row"},
            }
        ],
        HYPEROPT: {
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
            "eval_batch_size": None,
            "optimizer": {"type": "adadelta", "rho": 0.9, "eps": 1e-06, "lr": 1.0, "weight_decay": 0.0},
            "epochs": 100,
            "train_steps": None,
            "regularization_lambda": 0.0,
            "regularization_type": "l2",
            "should_shuffle": True,
            "learning_rate": 0.001,
            "batch_size": 128,
            "early_stop": 5,
            "steps_per_checkpoint": 0,
            "checkpoints_per_epoch": 0,
            "evaluate_training_set": True,
            "reduce_learning_rate_on_plateau": 0.0,
            "reduce_learning_rate_on_plateau_patience": 5,
            "reduce_learning_rate_on_plateau_rate": 0.5,
            "reduce_learning_rate_eval_metric": "loss",
            "reduce_learning_rate_eval_split": "training",
            "increase_batch_size_on_plateau": 0,
            "increase_batch_size_on_plateau_patience": 5,
            "increase_batch_size_on_plateau_rate": 2.0,
            "increase_batch_size_on_plateau_max": 512,
            "increase_batch_size_eval_metric": "loss",
            "increase_batch_size_eval_split": "training",
            "decay": False,
            "decay_steps": 10000,
            "decay_rate": 0.96,
            "staircase": False,
            "gradient_clipping": {"clipglobalnorm": 0.5, "clipnorm": None, "clipvalue": None},
            "validation_field": "combined",
            "validation_metric": "loss",
            "learning_rate_warmup_epochs": 1.0,
            "learning_rate_scaling": "linear",
        },
        PREPROCESSING: {
            "split": {},
            "undersample_majority": None,
            "oversample_minority": None,
            "sample_ratio": 1.0,
        },
        DEFAULTS: {
            "text": {
                PREPROCESSING: {
                    "tokenizer": "space_punct",
                    "pretrained_model_name_or_path": None,
                    "vocab_file": None,
                    "max_sequence_length": 256,
                    "most_common": 20000,
                    "padding_symbol": "<PAD>",
                    "unknown_symbol": "<UNK>",
                    "padding": "right",
                    "lowercase": True,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                }
            },
            "category": {
                PREPROCESSING: {
                    "most_common": 10000,
                    "lowercase": False,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                }
            },
            "set": {
                PREPROCESSING: {
                    "tokenizer": "space",
                    "most_common": 10000,
                    "lowercase": False,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                }
            },
            "bag": {
                PREPROCESSING: {
                    "tokenizer": "space",
                    "most_common": 10000,
                    "lowercase": False,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                }
            },
            "binary": {PREPROCESSING: {"missing_value_strategy": "fill_with_false"}},
            "number": {
                PREPROCESSING: {"missing_value_strategy": "fill_with_const", "fill_value": 0, "normalization": None}
            },
            "sequence": {
                PREPROCESSING: {
                    "max_sequence_length": 256,
                    "most_common": 20000,
                    "padding_symbol": "<PAD>",
                    "unknown_symbol": "<UNK>",
                    "padding": "right",
                    "tokenizer": "space",
                    "lowercase": False,
                    "vocab_file": None,
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "<UNK>",
                }
            },
            "timeseries": {
                PREPROCESSING: {
                    "timeseries_length_limit": 256,
                    "padding_value": 0,
                    "padding": "right",
                    "tokenizer": "space",
                    "missing_value_strategy": "fill_with_const",
                    "fill_value": "",
                }
            },
            "image": {
                PREPROCESSING: {
                    "missing_value_strategy": "backfill",
                    "in_memory": True,
                    "resize_method": "interpolate",
                    "scaling": "pixel_normalization",
                    "num_processes": 1,
                    "infer_image_num_channels": True,
                    "infer_image_dimensions": True,
                    "infer_image_max_height": 256,
                    "infer_image_max_width": 256,
                    "infer_image_sample_size": 100,
                }
            },
            "audio": {
                PREPROCESSING: {
                    "audio_file_length_limit_in_s": 7.5,
                    "missing_value_strategy": "backfill",
                    "in_memory": True,
                    "padding_value": 0,
                    "norm": None,
                    "type": "fbank",
                    "window_length_in_s": 0.04,
                    "window_shift_in_s": 0.02,
                    "num_fft_points": None,
                    "window_type": "hamming",
                    "num_filter_bands": 80,
                }
            },
            "h3": {PREPROCESSING: {"missing_value_strategy": "fill_with_const", "fill_value": 576495936675512319}},
            "date": {
                PREPROCESSING: {"missing_value_strategy": "fill_with_const", "fill_value": "", "datetime_format": None}
            },
            "vector": {PREPROCESSING: {"missing_value_strategy": "fill_with_const", "fill_value": ""}},
        },
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

    updated_config = merge_with_defaults(legacy_config_format)

    assert updated_config == expected_upgraded_format
