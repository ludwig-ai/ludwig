#! /usr/bin/env python
# coding=utf-8
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

import yaml

from ludwig.contrib import contrib_command, contrib_import
from ludwig.globals import LUDWIG_VERSION
from ludwig.hyperopt.run import hyperopt
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.horovod_utils import set_on_master, is_on_master
from ludwig.utils.misc_utils import check_which_model_definition
from ludwig.utils.print_utils import logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def hyperopt_cli(
        model_definition=None,
        model_definition_file=None,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        data_format=None,
        experiment_name="hyperopt",
        model_name="run",
        # model_load_path=None,
        # model_resume_path=None,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_save_eval_stats=True,
        skip_save_hyperopt_statistics=False,
        output_directory="results",
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=None,
        random_seed=default_random_seed,
        debug=False,
        **kwargs,
):
    model_definition = check_which_model_definition(model_definition,
                                                    model_definition_file)

    return hyperopt(
        model_definition=model_definition,
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        # model_load_path=model_load_path,
        # model_resume_path=model_resume_path,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        skip_save_hyperopt_statistics=skip_save_hyperopt_statistics,
        output_directory=output_directory,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug,
        **kwargs,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script searches for optimal Hyperparameters",
        prog="ludwig hyperopt",
        usage="%(prog)s [options]",
    )

    # -------------------
    # Hyperopt parameters
    # -------------------
    parser.add_argument(
        "-sshs",
        "--skip_save_hyperopt_statistics",
        help="skips saving hyperopt statistics file",
        action="store_true",
        default=False,
    )

    # ----------------------------
    # Experiment naming parameters
    # ----------------------------
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results",
        help="directory that contains the results",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="hyperopt",
        help="experiment name"
    )
    parser.add_argument(
        "--model_name", type=str, default="run", help="name for the model"
    )

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument(
        '--dataset',
        help='input data file path. '
             'If it has a split column, it will be used for splitting '
             '(0: train, 1: validation, 2: test), '
             'otherwise the dataset will be randomly split'
    )
    parser.add_argument('--training_set', help='input train data file path')
    parser.add_argument(
        '--validation_set', help='input validation data file path'
    )
    parser.add_argument('--test_set', help='input test data file path')

    parser.add_argument(
        '--training_set_metadata',
        help='input metadata JSON file path. An intermediate preprocess file '
             'containing the mappings of the input file created '
             'the first time a file is used, in the same directory '
             'with the same name and a .json extension'
    )

    parser.add_argument(
        '--data_format',
        help='format of the input data',
        default='auto',
        choices=['auto', 'csv', 'hdf5']
    )

    parser.add_argument(
        "-sspi",
        "--skip_save_processed_input",
        help="skips saving intermediate HDF5 and JSON files",
        action="store_true",
        default=False,
    )

    # ----------------
    # Model parameters
    # ----------------
    model_definition = parser.add_mutually_exclusive_group(required=True)
    model_definition.add_argument(
        "-md", "--model_definition", type=yaml.safe_load,
        help="model definition"
    )
    model_definition.add_argument(
        "-mdf",
        "--model_definition_file",
        help="YAML file describing the model. Ignores --model_hyperparameters",
    )

    parser.add_argument(
        "-mlp",
        "--model_load_path",
        help="path of a pretrained model to load as initialization",
    )
    parser.add_argument(
        "-mrp",
        "--model_resume_path",
        help="path of a the model directory to resume training of",
    )
    parser.add_argument(
        "-sstd",
        "--skip_save_training_description",
        action="store_true",
        default=False,
        help="disables saving the description JSON file",
    )
    parser.add_argument(
        "-ssts",
        "--skip_save_training_statistics",
        action="store_true",
        default=False,
        help="disables saving training statistics JSON file",
    )
    parser.add_argument(
        "-ssm",
        "--skip_save_model",
        action="store_true",
        default=False,
        help="disables saving weights each time the model imrpoves. "
             "By default Ludwig saves  weights after each epoch "
             "the validation metric imrpvoes, but  if the model is really big "
             "that can be time consuming if you do not want to keep "
             "the weights and just find out what performance can a model get "
             "with a set of hyperparameters, use this parameter to skip it",
    )
    parser.add_argument(
        "-ssp",
        "--skip_save_progress",
        action="store_true",
        default=False,
        help="disables saving weights after each epoch. By default ludwig saves "
             "weights after each epoch for enabling resuming of training, but "
             "if the model is really big that can be time consuming and will "
             "save twice as much space, use this parameter to skip it",
    )
    parser.add_argument(
        "-ssl",
        "--skip_save_log",
        action="store_true",
        default=False,
        help="disables saving TensorBoard logs. By default Ludwig saves "
             "logs for the TensorBoard, but if it is not needed turning it off "
             "can slightly increase the overall speed",
    )

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="a random seed that is going to be used anywhere there is a call "
             "to a random number generator: data splitting, parameter "
             "initialization and training set shuffling",
    )
    parser.add_argument(
        "-g", "--gpus", nargs="+", type=int, default=None,
        help="list of gpus to use"
    )
    parser.add_argument(
        '-gml',
        '--gpu_memory_limit',
        type=int,
        default=None,
        help='maximum memory in MB to allocate per GPU device'
    )
    parser.add_argument(
        "-uh",
        "--use_horovod",
        action="store_true",
        default=False,
        help="uses horovod for distributed training",
    )
    parser.add_argument(
        "-dbg",
        "--debug",
        action="store_true",
        default=False,
        help="enables debugging mode",
    )
    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    args = parser.parse_args(sys_argv)

    args.logging_level = logging_level_registry[args.logging_level]
    logging.getLogger('ludwig').setLevel(
        args.logging_level
    )
    global logger
    logger = logging.getLogger('ludwig.hyperopt')

    set_on_master(args.use_horovod)

    if is_on_master():
        print_ludwig("Hyperopt", LUDWIG_VERSION)

    hyperopt_cli(**vars(args))


if __name__ == "__main__":
    contrib_import()
    contrib_command("hyperopt", *sys.argv)
    cli(sys.argv[1:])
