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
import argparse
import logging
import sys
from typing import Union

import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, LOCAL, Backend, initialize_backend
from ludwig.contrib import contrib_command, contrib_import
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc_utils import check_which_config
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)


def preprocess_cli(
        preprocessing_config: dict = None,
        preprocessing_config_file: str = None,
        dataset: Union[str, dict, pd.DataFrame] = None,
        training_set: Union[str, dict, pd.DataFrame] = None,
        validation_set: Union[str, dict, pd.DataFrame] = None,
        test_set: Union[str, dict, pd.DataFrame] = None,
        training_set_metadata: Union[str, dict] = None,
        data_format: str = None,
        random_seed: int = default_random_seed,
        logging_level: int = logging.INFO,
        backend: Union[Backend, str] = None,
        debug: bool = False,
        **kwargs
) -> None:
    """*train* defines the entire training procedure used by Ludwig's
    internals. Requires most of the parameters that are taken into the model.
    Builds a full ludwig model and performs the training.

    :param config: (dict) config which defines the different
        parameters of the model, features, preprocessing and training.
    :param config_file: (str, default: `None`) the filepath string
        that specifies the config.  It is a yaml file.
    :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing the entire dataset to be used for training.
        If it has a split column, it will be used for splitting (0 for train,
        1 for validation, 2 for test), otherwise the dataset will be
        randomly split.
    :param training_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing training data.
    :param validation_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing validation data.
    :param test_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing test data.
    :param training_set_metadata: (Union[str, dict], default: `None`)
        metadata JSON file or loaded metadata.  Intermediate preprocessed
        structure containing the mappings of the input
        dataset created the first time an input file is used in the same
        directory with the same name and a '.meta.json' extension.
    :param data_format: (str, default: `None`) format to interpret data
        sources. Will be inferred automatically if not specified.  Valid
        formats are `'auto'`, `'csv'`, `'excel'`, `'feather'`,
        `'fwf'`, `'hdf5'` (cache file produced during previous training),
        `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
        `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
        `'stata'`, `'tsv'`.
    :param experiment_name: (str, default: `'experiment'`) name for
        the experiment.
    :param model_name: (str, default: `'run'`) name of the model that is
        being used.
    :param model_load_path: (str, default: `None`) if this is specified the
        loaded model will be used as initialization
        (useful for transfer learning).
    :param model_resume_path: (str, default: `None`) resumes training of
        the model from the path specified. The config is restored.
        In addition to config, training statistics, loss for each
        epoch and the state of the optimizer are restored such that
        training can be effectively continued from a previously interrupted
        training process.
    :param skip_save_training_description: (bool, default: `False`) disables
        saving the description JSON file.
    :param skip_save_training_statistics: (bool, default: `False`) disables
        saving training statistics JSON file.
    :param skip_save_model: (bool, default: `False`) disables
        saving model weights and hyperparameters each time the model
        improves. By default Ludwig saves model weights after each epoch
        the validation metric improves, but if the model is really big
        that can be time consuming. If you do not want to keep
        the weights and just find out what performance a model can get
        with a set of hyperparameters, use this parameter to skip it,
        but the model will not be loadable later on and the returned model
        will have the weights obtained at the end of training, instead of
        the weights of the epoch with the best validation performance.
    :param skip_save_progress: (bool, default: `False`) disables saving
        progress each epoch. By default Ludwig saves weights and stats
        after each epoch for enabling resuming of training, but if
        the model is really big that can be time consuming and will uses
        twice as much space, use this parameter to skip it, but training
        cannot be resumed later on.
    :param skip_save_log: (bool, default: `False`) disables saving
        TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
        but if it is not needed turning it off can slightly increase the
        overall speed.
    :param skip_save_processed_input: (bool, default: `False`) if input
        dataset is provided it is preprocessed and cached by saving an HDF5
        and JSON files to avoid running the preprocessing again. If this
        parameter is `False`, the HDF5 and JSON file are not saved.
    :param output_directory: (str, default: `'results'`) the directory that
        will contain the training statistics, TensorBoard logs, the saved
        model and the training progress files.
    :param gpus: (list, default: `None`) list of GPUs that are available
        for training.
    :param gpu_memory_limit: (int, default: `None`) maximum memory in MB to
        allocate per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow TensorFlow
        to use multithreading parallelism to improve performance at
        the cost of determinism.
    :param backend: (Union[Backend, str]) `Backend` or string name
        of backend to use to execute preprocessing / training steps.
    :param random_seed: (int: default: 42) random seed used for weights
        initialization, splits and any other random function.
    :param debug: (bool, default: `False) if `True` turns on `tfdbg` with
        `inf_or_nan` checks.
    :param logging_level: (int) Log level that will be sent to stderr.

    # Return

    :return: (`None`)
    """
    preprocessing_config = check_which_config(
        preprocessing_config,
        preprocessing_config_file
    )

    model = LudwigModel(
        config=preprocessing_config,
        logging_level=logging_level,
    )
    model.preprocess(
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        skip_save_processed_input=False,
        random_seed=random_seed,
        debug=debug,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script preprocess a dataset',
        prog='ludwig preprocess',
        usage='%(prog)s [options]'
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
        help='input metadata JSON file path. An intermediate preprocessed file '
             'containing the mappings of the input file created '
             'the first time a file is used, in the same directory '
             'with the same name and a .json extension'
    )

    parser.add_argument(
        '--data_format',
        help='format of the input data',
        default='auto',
        choices=['auto', 'csv', 'excel', 'feather', 'fwf', 'hdf5',
                 'html' 'tables', 'json', 'jsonl', 'parquet', 'pickle', 'sas',
                 'spss', 'stata', 'tsv']
    )

    # ----------------
    # Model parameters
    # ----------------
    preprocessing_def = parser.add_mutually_exclusive_group(required=True)
    preprocessing_def.add_argument(
        '-pc',
        '--preprocessing_config',
        type=yaml.safe_load,
        help='preproceesing config. '
             'Uses the same format of config, '
             'but ignores encoder specific parameters, '
             'decoder specific paramters, combiner and training parameters'
    )
    preprocessing_def.add_argument(
        '-pcf',
        '--preprocessing_config_file',
        help='YAML file describing the preprocessing. '
             'Ignores --preprocessing_config.'
             'Uses the same format of config, '
             'but ignores encoder specific parameters, '
             'decoder specific paramters, combiner and training parameters'
    )

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument(
        '-rs',
        '--random_seed',
        type=int,
        default=42,
        help='a random seed that is going to be used anywhere there is a call '
             'to a random number generator: data splitting, parameter '
             'initialization and training set shuffling'
    )
    parser.add_argument(
        "-b",
        "--backend",
        help='specifies backend to use for parallel / distributed execution, '
             'defaults to local execution or Horovod if called using horovodrun',
        choices=ALL_BACKENDS,
    )
    parser.add_argument(
        '-dbg',
        '--debug',
        action='store_true',
        default=False, help='enables debugging mode'
    )
    parser.add_argument(
        '-l',
        '--logging_level',
        default='info',
        help='the level of logging to use',
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    args = parser.parse_args(sys_argv)

    args.logging_level = logging_level_registry[args.logging_level]
    logging.getLogger('ludwig').setLevel(
        args.logging_level
    )
    global logger
    logger = logging.getLogger('ludwig.preprocess')

    args.backend = initialize_backend(args.backend)
    if args.backend.is_coordinator():
        print_ludwig('Preprocess', LUDWIG_VERSION)

    preprocess_cli(**vars(args))


if __name__ == '__main__':
    contrib_import()
    contrib_command("preprocess", *sys.argv)
    cli(sys.argv[1:])
