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
    SCHEDULER,
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
    # configuration with legacy parameters
    legacy_config_format = {
        "input_features": [
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
        "output_features": [
            {
                "type": "numerical",
                "name": "number_output_feature",
            },
        ],
        "training": {"eval_batch_size": 0},
        "hyperopt": {
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
        "input_features": [
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
                "column": "image_input_feature",
                "preprocessing": {},
                "proc_column": "image_input_feature_mZFLky",
                "tied": None,
                "encoder": "stacked_cnn",
                "conv_use_bias": True,
                "conv_layers": [
                    {
                        "num_filters": 32,
                        "pool_size": 2,
                        "pool_stride": 2,
                        "use_bias": False,
                    },
                    {
                        "num_filters": 64,
                        "pool_size": 2,
                        "pool_stride": 2,
                    },
                ],
            },
        ],
        "output_features": [
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
            "eval_batch_size": None,
            "optimizer": {"type": "adam", "betas": (0.9, 0.999), "eps": 1e-08},
            "epochs": 100,
            "regularization_lambda": 0,
            "regularization_type": "l2",
            "learning_rate": 0.001,
            "batch_size": 128,
            "early_stop": 5,
            "steps_per_checkpoint": 0,
            "reduce_learning_rate_on_plateau": 0,
            "reduce_learning_rate_on_plateau_patience": 5,
            "reduce_learning_rate_on_plateau_rate": 0.5,
            "increase_batch_size_on_plateau": 0,
            "increase_batch_size_on_plateau_patience": 5,
            "increase_batch_size_on_plateau_rate": 2,
            "increase_batch_size_on_plateau_max": 512,
            "decay": False,
            "decay_steps": 10000,
            "decay_rate": 0.96,
            "staircase": False,
            "gradient_clipping": None,
            "validation_field": "combined",
            "validation_metric": "loss",
            "learning_rate_warmup_epochs": 1,
        },
        "preprocessing": {
            "force_split": False,
            "split_probabilities": (0.7, 0.1, 0.2),
            "stratify": None,
            "undersample_majority": None,
            "oversample_minority": None,
            "sample_ratio": 1.0,
            "text": {
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
            },
            "category": {
                "most_common": 10000,
                "lowercase": False,
                "missing_value_strategy": "fill_with_const",
                "fill_value": "<UNK>",
            },
            "set": {
                "tokenizer": "space",
                "most_common": 10000,
                "lowercase": False,
                "missing_value_strategy": "fill_with_const",
                "fill_value": "<UNK>",
            },
            "bag": {
                "tokenizer": "space",
                "most_common": 10000,
                "lowercase": False,
                "missing_value_strategy": "fill_with_const",
                "fill_value": "<UNK>",
            },
            "binary": {"missing_value_strategy": "fill_with_false"},
            "number": {"missing_value_strategy": "fill_with_const", "fill_value": 0, "normalization": None},
            "sequence": {
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
            },
            "timeseries": {
                "timeseries_length_limit": 256,
                "padding_value": 0,
                "padding": "right",
                "tokenizer": "space",
                "missing_value_strategy": "fill_with_const",
                "fill_value": "",
            },
            "image": {
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
            },
            "audio": {
                "audio_file_length_limit_in_s": 7.5,
                "missing_value_strategy": "backfill",
                "in_memory": True,
                "padding_value": 0,
                "norm": None,
                "audio_feature": {
                    "type": "fbank",
                    "window_length_in_s": 0.04,
                    "window_shift_in_s": 0.02,
                    "num_filter_bands": 80,
                },
            },
            "h3": {"missing_value_strategy": "fill_with_const", "fill_value": 576495936675512319},
            "date": {"missing_value_strategy": "fill_with_const", "fill_value": "", "datetime_format": None},
            "vector": {"missing_value_strategy": "fill_with_const", "fill_value": ""},
        },
        "combiner": {"type": "concat"},
    }

    updated_config = merge_with_defaults(legacy_config_format)
    assert updated_config == expected_upgraded_format
