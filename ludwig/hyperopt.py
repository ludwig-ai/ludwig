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

from ludwig.constants import HYPEROPT
from ludwig.contrib import contrib_command
from ludwig.globals import LUDWIG_VERSION, is_on_master, set_on_master
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.utils.hyperopt_utils import get_build_hyperopt_strategy, get_build_hyperopt_executor, \
    update_hyperopt_params_with_defaults
from ludwig.utils.print_utils import logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def hyperopt(
        model_definition=None,
        model_definition_file=None,
        data_df=None,
        data_train_df=None,
        data_validation_df=None,
        data_test_df=None,
        data_csv=None,
        data_train_csv=None,
        data_validation_csv=None,
        data_test_csv=None,
        data_hdf5=None,
        data_train_hdf5=None,
        data_validation_hdf5=None,
        data_test_hdf5=None,
        train_set_metadata_json=None,
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
        skip_save_test_predictions=True,
        skip_save_test_statistics=True,
        output_directory="results",
        gpus=None,
        gpu_fraction=1.0,
        use_horovod=False,
        random_seed=default_random_seed,
        debug=False,
        **kwargs,
):
    # check for model_definition and model_definition_file
    if model_definition is None and model_definition_file is None:
        raise ValueError(
            "Either model_definition of model_definition_file have to be"
            "not None to initialize a LudwigModel"
        )
    if model_definition is not None and model_definition_file is not None:
        raise ValueError(
            "Only one between model_definition and "
            "model_definition_file can be provided"
        )

    # merge with default model definition to set defaults
    if model_definition_file is not None:
        with open(model_definition_file, "r") as def_file:
            model_definition = yaml.safe_load(def_file)
    model_definition = merge_with_defaults(model_definition)

    if HYPEROPT not in model_definition:
        raise ValueError(
            "Hyperopt Section not present in Model Definition"
        )

    hyperopt_params = model_definition["hyperopt"]
    update_hyperopt_params_with_defaults(hyperopt_params)

    strategy = hyperopt_params["strategy"]
    executor = hyperopt_params["executor"]
    parameters = hyperopt_params["parameters"]
    output_feature = hyperopt_params["output_feature"]
    metric = hyperopt_params["metric"]
    split = hyperopt_params["split"]

    hyperopt_strategy = get_build_hyperopt_strategy(
        strategy["type"]
    )(parameters, **strategy)
    hyperopt_executor = get_build_hyperopt_executor(
        executor["type"]
    )(hyperopt_strategy, output_feature, metric, split, **executor)

    results = hyperopt_executor.execute(
        model_definition,
        data_df=data_df,
        data_train_df=data_train_df,
        data_validation_df=data_validation_df,
        data_test_df=data_test_df,
        data_csv=data_csv,
        data_train_csv=data_train_csv,
        data_validation_csv=data_validation_csv,
        data_test_csv=data_test_csv,
        data_hdf5=data_hdf5,
        data_train_hdf5=data_train_hdf5,
        data_validation_hdf5=data_validation_hdf5,
        data_test_hdf5=data_test_hdf5,
        train_set_metadata_json=train_set_metadata_json,
        experiment_name=experiment_name,
        model_name=model_name,
        # model_load_path=None,
        # model_resume_path=None,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_test_predictions=skip_save_test_predictions,
        skip_save_test_statistics=skip_save_test_statistics,
        output_directory=output_directory,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug,
        **kwargs
    )

    return results


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script searches for Optimal Hyperparameters",
        prog="ludwig hyperopt",
        usage="%(prog)s [options]",
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
        "--experiment_name", type=str, default="hyperopt", help="experiment name"
    )
    parser.add_argument(
        "--model_name", type=str, default="run", help="name for the model"
    )

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument(
        "--data_csv",
        help="input data CSV file. "
             "If it has a split column, it will be used for splitting "
             "(0: train, 1: validation, 2: test), "
             "otherwise the dataset will be randomly split",
    )
    parser.add_argument("--data_train_csv", help="input train data CSV file")
    parser.add_argument("--data_validation_csv",
                        help="input validation data CSV file")
    parser.add_argument("--data_test_csv", help="input test data CSV file")

    parser.add_argument(
        "--data_hdf5",
        help="input data HDF5 file. It is an intermediate preprocess version of"
             " the input CSV created the first time a CSV file is used in the "
             "same directory with the same name and a hdf5 extension",
    )
    parser.add_argument(
        "--data_train_hdf5",
        help="input train data HDF5 file. It is an intermediate preprocess "
             "version of the input CSV created the first time a CSV file is "
             "used in the same directory with the same name and a hdf5 "
             "extension",
    )
    parser.add_argument(
        "--data_validation_hdf5",
        help="input validation data HDF5 file. It is an intermediate preprocess"
             " version of the input CSV created the first time a CSV file is "
             "used in the same directory with the same name and a hdf5 "
             "extension",
    )
    parser.add_argument(
        "--data_test_hdf5",
        help="input test data HDF5 file. It is an intermediate preprocess "
             "version of the input CSV created the first time a CSV file is "
             "used in the same directory with the same name and a hdf5 "
             "extension",
    )

    parser.add_argument(
        "--train_set_metadata_json",
        help="input metadata JSON file. It is an intermediate preprocess file "
             "containing the mappings of the input CSV created the first time a"
             " CSV file is used in the same directory with the same name and a "
             "json extension",
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
        "-md", "--model_definition", type=yaml.safe_load, help="model definition"
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
             "the validation measure imrpvoes, but  if the model is really big "
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
        "-g", "--gpus", nargs="+", type=int, default=None, help="list of gpus to use"
    )
    parser.add_argument(
        "-gf",
        "--gpu_fraction",
        type=float,
        default=1.0,
        help="fraction of gpu memory to initialize the process with",
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

    logging.getLogger("ludwig").setLevel(
        logging_level_registry[args.logging_level])
    set_on_master(args.use_horovod)

    if is_on_master():
        print_ludwig("Hyperopt", LUDWIG_VERSION)

    hyperopt(**vars(args))


if __name__ == "__main__":
    contrib_command("hyperopt", *sys.argv)
    cli(sys.argv[1:])
