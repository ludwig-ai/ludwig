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
    VALIDATION_METRIC,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.data.split import DEFAULT_PROBABILITIES, get_splitter
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema.config_object import Config
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.combiners.utils import combiner_registry
from ludwig.schema.utils import load_config_with_kwargs
from ludwig.utils.config_utils import get_default_encoder_or_decoder, get_defaults_section_for_feature_type
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.fs_utils import open_file
from ludwig.utils.misc_utils import get_from_registry, merge_dict, set_default_value, set_default_values
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)

default_random_seed = 42

# Still needed for preprocessing  TODO(Connor): Refactor ludwig/data/preprocessing to use schema
default_feature_specific_preprocessing_parameters = {name: preproc_sect.get_schema_cls()().preprocessing.to_dict() for
                                                     name, preproc_sect in input_type_registry.items()}

default_preprocessing_parameters = copy.deepcopy(default_feature_specific_preprocessing_parameters)
default_preprocessing_parameters.update(PreprocessingConfig().to_dict())


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


def merge_with_defaults(config: dict, config_obj: Config) -> dict:  # noqa: F821
    config = copy.deepcopy(config)
    _perform_sanity_checks(config)
    _set_feature_column(config)
    _set_proc_column(config)
    _merge_hyperopt_with_trainer(config)

    # ===== Defaults =====
    config[DEFAULTS] = config_obj.defaults.to_dict()

    # ===== Preprocessing =====
    config[PREPROCESSING] = config_obj.preprocessing.to_dict()
    splitter = get_splitter(**config[PREPROCESSING].get(SPLIT, {}))
    splitter.validate(config)

    # ===== Model Type =====
    config[MODEL_TYPE] = config_obj.model_type

    # ===== Trainer =====
    # Convert config dictionary into an instance of BaseTrainerConfig.
    config[TRAINER] = config_obj.trainer.to_dict()

    # ===== Input Features =====
    config[INPUT_FEATURES] = [getattr(config_obj.input_features, feat[NAME]).to_dict()
                              for feat in config[INPUT_FEATURES]]

    # ===== Combiner =====
    config[COMBINER] = config_obj.combiner.to_dict()

    # ===== Output features =====
    config[OUTPUT_FEATURES] = [getattr(config_obj.output_features, feat[NAME]).to_dict()
                               for feat in config[OUTPUT_FEATURES]]

    # ===== Hyperpot =====
    if HYPEROPT in config:
        set_default_value(config[HYPEROPT][EXECUTOR], TYPE, RAY)
    return config


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
