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
from typing import Dict

import yaml

from ludwig.constants import (
    EXECUTOR,
    HYPEROPT,
    RAY,
    TRAINER,
    TYPE,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.features.feature_registries import input_type_registry
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema import validate_config
from ludwig.schema.config_object import Config
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.utils.backward_compatibility import upgrade_to_latest_version
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.fs_utils import open_file
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)

default_random_seed = 42

# Still needed for preprocessing  TODO(Connor): Refactor ludwig/data/preprocessing to use schema
default_feature_specific_preprocessing_parameters = {
    name: preproc_sect.get_schema_cls()().preprocessing.to_dict() for name, preproc_sect in input_type_registry.items()
}

default_preprocessing_parameters = copy.deepcopy(default_feature_specific_preprocessing_parameters)
default_preprocessing_parameters.update(PreprocessingConfig().to_dict())


def merge_hyperopt_with_trainer(config: dict) -> None:
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


def set_hyperopt_defaults(config: dict) -> Dict[str, any]:
    """
    This function is intended to set the defaults for hyperopt on the user defined config. This is a temporary function
    that should be removed once Hyperopt has been reconfigured to use the config object instead of a config dictionary.

    Args:
        config: User defined config dictionary

    Returns:
        Updated user defined config dictionary
    """

    merge_hyperopt_with_trainer(config)

    if HYPEROPT in config:
        set_default_value(config[HYPEROPT][EXECUTOR], TYPE, RAY)

    return config


def render_config(config=None, output=None, **kwargs):
    upgraded_config = upgrade_to_latest_version(config)
    output_config = Config(upgraded_config).get_config_dict()
    validate_config(output_config)

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
