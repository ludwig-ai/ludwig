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
import warnings
from typing import Any, Dict

import yaml

from ludwig.constants import (
    BINARY,
    CATEGORY,
    COLUMN,
    DROP_ROW,
    HYPEROPT,
    NAME,
    NUMBER,
    PREPROCESSING,
    PROC_COLUMN,
    TRAINER,
    TYPE,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.features.feature_registries import base_type_registry, input_type_registry, output_type_registry
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.misc_utils import get_from_registry, merge_dict, set_default_value
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)

default_random_seed = 42

default_preprocessing_force_split = False
default_preprocessing_split_probabilities = (0.7, 0.1, 0.2)
default_preprocessing_stratify = None
default_preprocessing_undersample_majority = None
default_preprocessing_oversample_minority = None

default_preprocessing_parameters = {
    "force_split": default_preprocessing_force_split,
    "split_probabilities": default_preprocessing_split_probabilities,
    "stratify": default_preprocessing_stratify,
    "undersample_majority": default_preprocessing_undersample_majority,
    "oversample_minority": default_preprocessing_oversample_minority,
}
default_preprocessing_parameters.update(
    {name: base_type.preprocessing_defaults() for name, base_type in base_type_registry.items()}
)

default_combiner_type = "concat"


def _upgrade_deprecated_fields(config: Dict[str, Any]):
    if "training" in config:
        warnings.warn('Config section "training" renamed to "trainer" and will be removed in v0.6', DeprecationWarning)
        config[TRAINER] = config["training"]
        del config["training"]

    for feature in config.get("input_features", []) + config.get("output_features", []):
        if feature.get(TYPE) == "numerical":
            warnings.warn(
                'Feature type "numerical" renamed to "number" and will be removed in v0.6', DeprecationWarning
            )
            feature[TYPE] = NUMBER

    if HYPEROPT in config and "parameters" in config[HYPEROPT]:
        hparams = config[HYPEROPT]["parameters"]
        for k, v in list(hparams.items()):
            substr = "training."
            if k.startswith(substr):
                warnings.warn(
                    'Config section "training" renamed to "trainer" and will be removed in v0.6', DeprecationWarning
                )
                hparams["trainer." + k[len(substr) :]] = v
                del hparams[k]


def _perform_sanity_checks(config):
    assert "input_features" in config, "config does not define any input features"

    assert "output_features" in config, "config does not define any output features"

    assert isinstance(config["input_features"], list), (
        "Ludwig expects input features in a list. Check your model " "config format"
    )

    assert isinstance(config["output_features"], list), (
        "Ludwig expects output features in a list. Check your model " "config format"
    )

    assert len(config["input_features"]) > 0, "config needs to have at least one input feature"

    assert len(config["output_features"]) > 0, "config needs to have at least one output feature"

    if TRAINER in config:
        assert isinstance(config[TRAINER], dict), (
            "There is an issue while reading the training section of the "
            "config. The parameters are expected to be"
            "read as a dictionary. Please check your config format."
        )

    if "preprocessing" in config:
        assert isinstance(config["preprocessing"], dict), (
            "There is an issue while reading the preprocessing section of the "
            "config. The parameters are expected to be read"
            "as a dictionary. Please check your config format."
        )

    if "combiner" in config:
        assert isinstance(config["combiner"], dict), (
            "There is an issue while reading the combiner section of the "
            "config. The parameters are expected to be read"
            "as a dictionary. Please check your config format."
        )


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

    scheduler = config["hyperopt"].get("sampler", {}).get("scheduler")
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

    # At most one of max_t and epochs may be specified by the user, and we set them to be equal to
    # ensure that Ludwig does not stop training before the scheduler has finished the trial, unless
    # max_t is in time_total_s, in which case we set epochs very high to continue train until stopped.
    max_t = scheduler.get("max_t")
    time_attr = scheduler.get("time_attr")
    epochs = trainer.get("epochs")
    if max_t is not None and epochs is not None and max_t != epochs and time_attr != "time_total_s":
        raise ValueError(
            "Cannot set trainer parameter `epochs` when using a hyperopt scheduler with `max_t`. "
            "Unset one of these parameters in your config."
        )
    elif max_t is not None:
        if time_attr == "time_total_s":
            trainer["epochs"] = sys.maxsize  # essentially continue training until stopped
        else:
            trainer["epochs"] = max_t
    elif epochs is not None:
        scheduler["max_t"] = epochs


def merge_with_defaults(config):
    config = copy.deepcopy(config)
    _upgrade_deprecated_fields(config)
    _perform_sanity_checks(config)
    _set_feature_column(config)
    _set_proc_column(config)
    _merge_hyperopt_with_trainer(config)

    # ===== Preprocessing =====
    config["preprocessing"] = merge_dict(default_preprocessing_parameters, config.get("preprocessing", {}))

    stratify = config["preprocessing"]["stratify"]
    if stratify is not None:
        features = config["input_features"] + config["output_features"]
        feature_names = {f[COLUMN] for f in features}
        if stratify not in feature_names:
            logger.warning("Stratify is not among the features. " "Cannot establish if it is a binary or category")
        elif [f for f in features if f[COLUMN] == stratify][0][TYPE] not in {BINARY, CATEGORY}:
            raise ValueError("Stratify feature must be binary or category")

    # ===== Training =====
    set_default_value(
        config[TRAINER],
        "validation_metric",
        output_type_registry[config["output_features"][0][TYPE]].default_validation_metric,
    )

    # ===== Input Features =====
    for input_feature in config["input_features"]:
        get_from_registry(input_feature[TYPE], input_type_registry).populate_defaults(input_feature)

    # ===== Combiner =====
    set_default_value(config, "combiner", {TYPE: default_combiner_type})

    # ===== Output features =====
    for output_feature in config["output_features"]:
        get_from_registry(output_feature[TYPE], output_type_registry).populate_defaults(output_feature)

        # By default, drop rows with missing output features
        set_default_value(output_feature, PREPROCESSING, {})
        set_default_value(output_feature[PREPROCESSING], "missing_value_strategy", DROP_ROW)

    return config


def render_config(config=None, output=None, **kwargs):
    output_config = merge_with_defaults(config)
    if output is None:
        print(yaml.safe_dump(output_config, None, sort_keys=False))
    else:
        with open(output, "w") as f:
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
