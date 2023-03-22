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

import yaml

from ludwig.api_annotations import DeveloperAPI
from ludwig.contrib import add_contrib_callback_args
from ludwig.features.feature_registries import get_input_type_registry
from ludwig.globals import LUDWIG_VERSION
from ludwig.schema.model_config import ModelConfig
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.fs_utils import open_file
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)

default_random_seed = 42

# Still needed for preprocessing  TODO(Connor): Refactor ludwig/data/preprocessing to use schema
# TODO(travis): remove this, make type a protected string for each subclass
default_feature_specific_preprocessing_parameters = {
    name: preproc_sect.get_schema_cls()(name="__tmp__", type=name).preprocessing.to_dict()
    for name, preproc_sect in get_input_type_registry().items()
}

default_training_preprocessing_parameters = copy.deepcopy(default_feature_specific_preprocessing_parameters)
default_training_preprocessing_parameters.update(PreprocessingConfig().to_dict())

default_prediction_preprocessing_parameters = copy.deepcopy(default_feature_specific_preprocessing_parameters)


@DeveloperAPI
def render_config(config=None, output=None, **kwargs):
    upgraded_config = upgrade_config_dict_to_latest_version(config)
    output_config = ModelConfig.from_dict(upgraded_config).to_dict()

    if output is None:
        print(yaml.safe_dump(output_config, None, sort_keys=False))
    else:
        with open_file(output, "w") as f:
            yaml.safe_dump(output_config, f, sort_keys=False)


@DeveloperAPI
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
