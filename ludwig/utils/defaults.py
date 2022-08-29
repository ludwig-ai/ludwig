#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import copy
import logging
import sys
from dataclasses import asdict
from typing import Any, Dict

import yaml

from ludwig.constants import (
    COLUMN,
    COMBINER,
    DECODER,
    DEFAULTS,
    DROP_ROW,
    ENCODER,
    EXECUTOR,
    HYPEROPT,
    INPUT_FEATURES,
    LOSS,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_TYPE,
    NAME,
    OUTPUT_FEATURES,
    PREPROCESSING,
    PROC_COLUMN,
    RAY,
    SPLIT,
    TRAINER,
    TYPE,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.data.split import DEFAULT_PROBABILITIES, get_splitter
from ludwig.features.feature_registries import base_type_registry, input_type_registry, output_type_registry
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema.combiners.utils import combiner_registry
from ludwig.schema.utils import load_config_with_kwargs, load_trainer_with_kwargs
from ludwig.utils.config_utils import get_default_encoder_or_decoder, get_defaults_section_for_feature_type
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.fs_utils import open_file
from ludwig.utils.misc_utils import get_from_registry, merge_dict, set_default_value, set_default_values
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)

default_random_seed = 42

BASE_PREPROCESSING_SPLIT_CONFIG = {"type": "random", "probabilities": list(DEFAULT_PROBABILITIES)}
base_preprocessing_parameters = {
    "split": BASE_PREPROCESSING_SPLIT_CONFIG,
    "undersample_majority": None,
    "oversample_minority": None,
    "sample_ratio": 1.0,
}

default_feature_specific_preprocessing_parameters = {
    name: base_type.preprocessing_defaults() for name, base_type in base_type_registry.items()
}

default_preprocessing_parameters = copy.deepcopy(default_feature_specific_preprocessing_parameters)
default_preprocessing_parameters.update(base_preprocessing_parameters)

default_model_type = MODEL_ECD

default_combiner_type = "concat"


def _perform_sanity_checks(config):
    assert INPUT_FEATURES in config, "config does not define any input features"

    assert OUTPUT_FEATURES in config, "config does not define any output features"

    assert isinstance(config[INPUT_FEATURES], list), (
        "Ludwig expects input features in a list. Check your model " "config format"
    )

    assert isinstance(config[OUTPUT_FEATURES], list), (
        "Ludwig expects output features in a list. Check your model " "config format"
    )

    assert len(config[INPUT_FEATURES]) > 0, "config needs to have at least one input feature"

    assert len(config[OUTPUT_FEATURES]) > 0, "config needs to have at least one output feature"

    if TRAINER in config:
        assert isinstance(config[TRAINER], dict), (
            "There is an issue while reading the training section of the "
            "config. The parameters are expected to be"
            "read as a dictionary. Please check your config format."
        )

    if PREPROCESSING in config:
        assert isinstance(config[PREPROCESSING], dict), (
            "There is an issue while reading the preprocessing section of the "
            "config. The parameters are expected to be read"
            "as a dictionary. Please check your config format."
        )

    if COMBINER in config:
        assert isinstance(config[COMBINER], dict), (
            "There is an issue while reading the combiner section of the "
            "config. The parameters are expected to be read"
            "as a dictionary. Please check your config format."
        )

    if MODEL_TYPE in config:
        assert isinstance(
            config[MODEL_TYPE], str
        ), "Ludwig expects model type as a string. Please check your model config format."

    if DEFAULTS in config:
        for feature_type in config.get(DEFAULTS).keys():
            # output_feature_types is a subset of input_feature_types so just check input_feature_types
            assert feature_type in set(
                input_type_registry.keys()
            ), f"""Defaults specified for `{feature_type}` but `{feature_type}` is
                not a feature type recognised by Ludwig."""

            for feature_type_param in config.get(DEFAULTS).get(feature_type).keys():
                assert feature_type_param in {
                    PREPROCESSING,
                    ENCODER,
                    DECODER,
                    LOSS,
                }, f"""`{feature_type_param}` is not a recognised subsection of Ludwig defaults. Valid default config
                 sections are {PREPROCESSING}, {ENCODER}, {DECODER} and {LOSS}."""


def _set_feature_column(config: dict) -> None:
    for feature in config["input_features"] + config["output_features"]:
        if COLUMN not in feature:
            feature[COLUMN] = feature[NAME]


def _set_proc_column(config: dict) -> None:
    for feature in config["input_features"] + config["output_features"]:
        if PROC_COLUMN not in feature:
            feature[PROC_COLUMN] = compute_feature_hash(feature)


def _merge_hyperopt_with_trainer(config: dict) -> None:
    if "hyperopt" not in config:
        return

    scheduler = config["hyperopt"].get("executor", {}).get("scheduler")
    if not scheduler:
        return

    if TRAINER not in config:
        config[TRAINER] = {}

    # Disable early stopping when using a scheduler. We achieve this by setting the parameter
    # to -1, which ensures the condition to apply early stopping is never met.
    trainer = config[TRAINER]
    early_stop = trainer.get("early_stop")
    if early_stop is not None and early_stop != -1:
        raise ValueError(
            "Cannot set trainer parameter `early_stop` when using a hyperopt scheduler. "
            "Unset this parameter in your config."
        )
    trainer["early_stop"] = -1

    max_t = scheduler.get("max_t")
    time_attr = scheduler.get("time_attr")
    epochs = trainer.get("epochs")
    if max_t is not None:
        if time_attr == "time_total_s":
            if epochs is None:
                trainer["epochs"] = sys.maxsize  # continue training until time limit hit
            # else continue training until either time or trainer epochs limit hit
        elif epochs is not None and epochs != max_t:
            raise ValueError(
                "Cannot set trainer `epochs` when using hyperopt scheduler w/different training_iteration `max_t`. "
                "Unset one of these parameters in your config or make sure their values match."
            )
        else:
            trainer["epochs"] = max_t  # run trainer until scheduler epochs limit hit
    elif epochs is not None:
        scheduler["max_t"] = epochs  # run scheduler until trainer epochs limit hit


def update_feature_from_defaults(config: Dict[str, Any], feature_dict: Dict[str, Any], config_feature_group: str):
    """Updates feature_dict belonging to an input or output feature using global encoder, decoder and loss related
    default parameters specified in the Ludwig config.

    :param config: Ludwig configuration containing parameters for different sections, including global default
        parameters for preprocessing, encoder, decoder and loss.
    :type config: dict[str, any]
    :param feature_dict: Underlying config for the specific input/output feature. This may be updated with values
        from the global defaults specified in config.
    :type feature_dict: dict[str, any]
    :param config_feature_group: Indicates whether the feature is an input feature or output feature (can be either of
        `input_features` or `output_features`).
    :type config_feature_group: str
    """
    parameter = ENCODER if config_feature_group == INPUT_FEATURES else DECODER
    registry_type = input_type_registry if config_feature_group == INPUT_FEATURES else output_type_registry

    default_params_for_feature_type = get_defaults_section_for_feature_type(
        feature_dict[TYPE], config[DEFAULTS], parameter
    )

    # Update input feature encoder or output feature decoder if it is specified in global defaults
    # TODO(#2125): This code block needs some refactoring.
    if TYPE in default_params_for_feature_type:
        # Only update encoder or decoder if the feature isn't already using a default encoder or decoder
        default_encoder_or_decoder = get_default_encoder_or_decoder(feature_dict, config_feature_group)
        if default_params_for_feature_type[TYPE] != default_encoder_or_decoder:
            # Update type and populate defaults for the encoder or decoder type
            feature_dict[parameter] = default_params_for_feature_type
            get_from_registry(feature_dict[TYPE], registry_type).populate_defaults(feature_dict)
        # Make a copy of default encoder or decoder parameters without the type key.
        default_params_for_feature_type = copy.deepcopy(default_params_for_feature_type)
        default_params_for_feature_type.pop(TYPE, None)

    # Update encoder or decoder with other encoder/decoder related parameters
    feature_dict.update(merge_dict(feature_dict, default_params_for_feature_type))

    # Update loss parameters for output feature from global defaults
    if parameter == DECODER:
        default_loss_params_for_feature_type = get_defaults_section_for_feature_type(
            feature_dict[TYPE], config[DEFAULTS], LOSS
        )
        feature_dict[LOSS].update(merge_dict(feature_dict[LOSS], default_loss_params_for_feature_type))


def merge_with_defaults(config: dict) -> dict:  # noqa: F821
    config = copy.deepcopy(config)
    _perform_sanity_checks(config)
    _set_feature_column(config)
    _set_proc_column(config)
    _merge_hyperopt_with_trainer(config)

    # ===== Defaults =====
    if DEFAULTS not in config:
        config[DEFAULTS] = dict()

    # Update defaults with the default feature specific preprocessing parameters
    for feature_type, preprocessing_defaults in default_feature_specific_preprocessing_parameters.items():
        # Create a new key with feature type if defaults is empty
        if feature_type not in config.get(DEFAULTS):
            if PREPROCESSING in preprocessing_defaults:
                config[DEFAULTS][feature_type] = preprocessing_defaults
            else:
                config[DEFAULTS][feature_type] = {PREPROCESSING: preprocessing_defaults}
        # Feature type exists but preprocessing hasn't been specified
        elif PREPROCESSING not in config[DEFAULTS][feature_type]:
            config[DEFAULTS][feature_type][PREPROCESSING] = preprocessing_defaults
        # Preprocessing parameters exist for feature type, update defaults with parameters from config
        else:
            config[DEFAULTS][feature_type][PREPROCESSING].update(
                merge_dict(preprocessing_defaults, config[DEFAULTS][feature_type][PREPROCESSING])
            )

    # ===== Preprocessing =====
    config[PREPROCESSING] = merge_dict(base_preprocessing_parameters, config.get(PREPROCESSING, {}))
    splitter = get_splitter(**config[PREPROCESSING].get(SPLIT, {}))
    splitter.validate(config)

    # ===== Model Type =====
    set_default_value(config, MODEL_TYPE, default_model_type)

    # ===== Training =====
    # Convert config dictionary into an instance of BaseTrainerConfig.
    full_trainer_config, _ = load_trainer_with_kwargs(config[MODEL_TYPE], config[TRAINER] if TRAINER in config else {})
    config[TRAINER] = asdict(full_trainer_config)

    set_default_value(
        config[TRAINER],
        "validation_metric",
        output_type_registry[config[OUTPUT_FEATURES][0][TYPE]].default_validation_metric,
    )

    # ===== Input Features =====
    for input_feature in config[INPUT_FEATURES]:
        if config[MODEL_TYPE] == MODEL_GBM:
            set_default_values(input_feature, {ENCODER: {TYPE: "passthrough"}})
            remove_ecd_params(input_feature)
        get_from_registry(input_feature[TYPE], input_type_registry).populate_defaults(input_feature)

        # Update encoder parameters for output feature from global defaults
        update_feature_from_defaults(config, input_feature, INPUT_FEATURES)

    # ===== Combiner =====
    set_default_value(config, COMBINER, {TYPE: default_combiner_type})
    full_combiner_config, _ = load_config_with_kwargs(
        combiner_registry[config[COMBINER][TYPE]].get_schema_cls(), config[COMBINER]
    )
    config[COMBINER].update(asdict(full_combiner_config))

    # ===== Output features =====
    for output_feature in config[OUTPUT_FEATURES]:
        if config[MODEL_TYPE] == MODEL_GBM:
            set_default_values(output_feature, {DECODER: {TYPE: "passthrough"}})
            remove_ecd_params(output_feature)
        get_from_registry(output_feature[TYPE], output_type_registry).populate_defaults(output_feature)

        # By default, drop rows with missing output features
        set_default_value(output_feature, PREPROCESSING, {})
        set_default_value(output_feature[PREPROCESSING], "missing_value_strategy", DROP_ROW)

        # Update decoder and loss related parameters for output feature from global defaults
        update_feature_from_defaults(config, output_feature, OUTPUT_FEATURES)

    # ===== Hyperpot =====
    if HYPEROPT in config:
        set_default_value(config[HYPEROPT][EXECUTOR], TYPE, RAY)
    return config


def remove_ecd_params(feature):
    feature.pop("tied", None)
    feature.pop("fc_layers", None)
    feature.pop("num_layers", None)
    feature.pop("output_size", None)
    feature.pop("use_bias", None)
    feature.pop("weights_initializer", None)
    feature.pop("bias_initializer", None)
    feature.pop("norm", None)
    feature.pop("norm_params", None)
    feature.pop("activation", None)
    feature.pop("dropout", None)
    feature.pop("embedding_size", None)
    feature.pop("embeddings_on_cpu", None)
    feature.pop("pretrained_embeddings", None)
    feature.pop("embeddings_trainable", None)
    feature.pop("embedding_initializer", None)
    # decoder params
    feature.pop("reduce_input", None)
    feature.pop("dependencies", None)
    feature.pop("reduce_dependencies", None)
    feature.pop("loss", None)
    feature.pop("num_fc_layers", None)
    feature.pop("threshold", None)
    feature.pop("clip", None)
    feature.pop("top_k", None)


def render_config(config=None, output=None, **kwargs):
    output_config = merge_with_defaults(config)
    if output is None:
        print(yaml.safe_dump(output_config, None, sort_keys=False))
    else:
        with open_file(output, "w") as f:
            yaml.safe_dump(output_config, f, sort_keys=False)


def cli_render_config(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script renders the full config from a user config.",
        prog="ludwig render_config",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=load_yaml,
        help="Path to the YAML file containing the model configuration",
    )
    parser.add_argument(
        "-cs",
        "--config_str",
        dest="config",
        type=load_config_from_str,
        help="JSON or YAML serialized string of the model configuration",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output rendered YAML config path",
        required=False,
    )

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("render_config", *sys_argv)

    print_ludwig("Render Config", LUDWIG_VERSION)
    render_config(**vars(args))
