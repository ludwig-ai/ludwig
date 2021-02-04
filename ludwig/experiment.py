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
import os
import sys
from typing import List, Union

import pandas as pd
import yaml

from ludwig.api import LudwigModel, kfold_cross_validate
from ludwig.backend import ALL_BACKENDS, LOCAL, Backend, initialize_backend
from ludwig.constants import FULL, TEST, TRAINING, VALIDATION
from ludwig.contrib import contrib_command, contrib_import
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc_utils import check_which_config
from ludwig.utils.print_utils import logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def experiment_cli(
        config: dict,
        config_file: str = None,
        dataset: Union[str, dict, pd.DataFrame] = None,
        training_set: Union[str, dict, pd.DataFrame] = None,
        validation_set: Union[str, dict, pd.DataFrame] = None,
        test_set: Union[str, dict, pd.DataFrame] = None,
        training_set_metadata: Union[str, dict] = None,
        data_format: str = None,
        experiment_name: str = 'experiment',
        model_name: str = 'run',
        model_load_path: str = None,
        model_resume_path: str = None,
        eval_split: str = TEST,
        skip_save_training_description: bool = False,
        skip_save_training_statistics: bool = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        skip_save_processed_input: bool = False,
        skip_save_unprocessed_output: bool = False,
        skip_save_predictions: bool = False,
        skip_save_eval_stats: bool = False,
        skip_collect_predictions: bool = False,
        skip_collect_overall_stats: bool = False,
        output_directory: str = 'results',
        gpus: Union[str, int, List[int]] = None,
        gpu_memory_limit: int = None,
        allow_parallel_threads: bool = True,
        backend: Union[Backend, str] = None,
        random_seed: int = default_random_seed,
        debug: bool = False,
        logging_level: int = logging.INFO,
        **kwargs
):
    """Trains a model on a dataset's training and validation splits and
    uses it to predict on the test split.
    It saves the trained model and the statistics of training and testing.

    # Inputs

    :param config: (dict) config which defines the different
        parameters of the model, features, preprocessing and training.
    :param config_file: (str, default: `None`) the filepath string
        that specifies the config.  It is a yaml file.
    :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing the entire dataset to be used in the experiment.
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
        In addition to config, training statistics and loss for
        epoch and the state of the optimizer are restored such that
        training can be effectively continued from a previously interrupted
        training process.
    :param eval_split: (str, default: `test`) split on which
        to perform evaluation. Valid values are `training`, `validation`
        and `test`.
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
    :param skip_save_unprocessed_output: (bool, default: `False`) by default
        predictions and their probabilities are saved in both raw
        unprocessed numpy files containing tensors and as postprocessed
        CSV files (one for each output feature). If this parameter is True,
        only the CSV ones are saved and the numpy ones are skipped.
    :param skip_save_predictions: (bool, default: `False`) skips saving test
        predictions CSV files
    :param skip_save_eval_stats: (bool, default: `False`) skips saving test
        statistics JSON file
   :param skip_collect_predictions: (bool, default: `False`) skips
        collecting post-processed predictions during eval.
    :param skip_collect_overall_stats: (bool, default: `False`) skips
        collecting overall stats during eval.
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
    :return: (Tuple[LudwigModel, dict, dict, tuple, str)) `(model, evaluation_statistics, training_statistics, preprocessed_data, output_directory)`
        `model` LudwigModel instance
        `evaluation_statistics` dictionary with evaluation performance
            statistics on the test_set,
        `training_statistics` is a dictionary of training statistics for
            each output
        feature containing loss and metrics values for each epoch,
        `preprocessed_data` tuple containing preprocessed
        `(training_set, validation_set, test_set)`, `output_directory`
        filepath string to where results are stored.

    """
    backend = initialize_backend(backend)

    config = check_which_config(config,
                                config_file)

    if model_load_path:
        model = LudwigModel.load(model_load_path)
    else:
        model = LudwigModel(
            config=config,
            logging_level=logging_level,
            backend=backend,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
        )
    (
        eval_stats,
        train_stats,
        preprocessed_data,
        output_directory
    ) = model.experiment(
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        model_resume_path=model_resume_path,
        eval_split=eval_split,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        skip_collect_predictions=skip_collect_predictions,
        skip_collect_overall_stats=skip_collect_overall_stats,
        output_directory=output_directory,
        random_seed=random_seed,
        debug=debug,
    )

    return model, eval_stats, train_stats, preprocessed_data, output_directory


def kfold_cross_validate_cli(
        k_fold,
        config=None,
        config_file=None,
        dataset=None,
        data_format=None,
        output_directory='results',
        random_seed=default_random_seed,
        skip_save_k_fold_split_indices=False,
        **kwargs
):
    """Wrapper function to performs k-fold cross validation.

    # Inputs
    :param k_fold: (int) number of folds to create for the cross-validation
    :param config: (dict, default: None) a dictionary containing
            information needed to build a model. Refer to the [User Guide]
           (http://ludwig.ai/user_guide/#model-config) for details.
    :param config_file: (string, optional, default: `None`) path to
           a YAML file containing the config. If available it will be
           used instead of the config dict.
    :param data_csv: (string, default: None)
    :param output_directory: (string, default: 'results')
    :param random_seed: (int) Random seed used k-fold splits.
    :param skip_save_k_fold_split_indices: (boolean, default: False) Disables
            saving k-fold split indices

    :return: None
    """

    if config is None and config_file is None:
        raise ValueError(
            "No config is provided 'config' or "
            "'config_file' must be provided."
        )
    elif config is not None and config_file is not None:
        raise ValueError(
            "Cannot specify both 'config' and 'config_file'"
            ", proivde only one of the parameters."
        )

    (kfold_cv_stats,
     kfold_split_indices) = kfold_cross_validate(
        k_fold,
        config=config if config is not None else
        config_file,
        dataset=dataset,
        data_format=data_format,
        output_directory=output_directory,
        random_seed=random_seed
    )

    # save k-fold cv statistics
    save_json(os.path.join(output_directory, 'kfold_training_statistics.json'),
              kfold_cv_stats)

    # save k-fold split indices
    if not skip_save_k_fold_split_indices:
        save_json(os.path.join(output_directory, 'kfold_split_indices.json'),
                  kfold_split_indices)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script trains and evaluates a model',
        prog='ludwig experiment',
        usage='%(prog)s [options]'
    )

    # ----------------------------
    # Experiment naming parameters
    # ----------------------------
    parser.add_argument(
        '--output_directory',
        type=str,
        default='results',
        help='directory that contains the results'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='experiment',
        help='experiment name'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='run',
        help='name for the model'
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

    parser.add_argument(
        '-es',
        '--eval_split',
        default=TEST,
        choices=[TRAINING, VALIDATION, TEST, FULL],
        help='the split to evaluate the model on'
    )

    parser.add_argument(
        '-sspi',
        '--skip_save_processed_input',
        help='skips saving intermediate HDF5 and JSON files',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-ssuo',
        '--skip_save_unprocessed_output',
        help='skips saving intermediate NPY output files',
        action='store_true',
        default=False
    )

    # -----------------
    # K-fold parameters
    # -----------------
    parser.add_argument(
        '-kf',
        '--k_fold',
        type=int,
        default=None,
        help='number of folds for a k-fold cross validation run '
    )
    parser.add_argument(
        '-skfsi',
        '--skip_save_k_fold_split_indices',
        action='store_true',
        default=False,
        help='disables saving indices generated to split training data set '
             'for the k-fold cross validation run, but if it is not needed '
             'turning it off can slightly increase the overall speed'
    )

    # ----------------
    # Model parameters
    # ----------------
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument(
        '-c',
        '--config',
        type=yaml.safe_load,
        help='config'
    )
    config.add_argument(
        '-cf',
        '--config_file',
        help='YAML file describing the model. Ignores --model_hyperparameters'
    )

    parser.add_argument(
        '-mlp',
        '--model_load_path',
        help='path of a pretrained model to load as initialization'
    )
    parser.add_argument(
        '-mrp',
        '--model_resume_path',
        help='path of the model directory to resume training of'
    )
    parser.add_argument(
        '-sstd',
        '--skip_save_training_description',
        action='store_true',
        default=False,
        help='disables saving the description JSON file'
    )
    parser.add_argument(
        '-ssts',
        '--skip_save_training_statistics',
        action='store_true',
        default=False,
        help='disables saving training statistics JSON file'
    )
    parser.add_argument(
        '-sstp',
        '--skip_save_predictions',
        help='skips saving test predictions CSV files',
        action='store_true', default=False
    )
    parser.add_argument(
        '-sstes',
        '--skip_save_eval_stats',
        help='skips saving eval statistics JSON file',
        action='store_true', default=False
    )
    parser.add_argument(
        '-ssm',
        '--skip_save_model',
        action='store_true',
        default=False,
        help='disables saving model weights and hyperparameters each time '
             'the model improves. '
             'By default Ludwig saves model weights after each epoch '
             'the validation metric imprvoes, but if the model is really big '
             'that can be time consuming. If you do not want to keep '
             'the weights and just find out what performance a model can get '
             'with a set of hyperparameters, use this parameter to skip it,'
             'but the model will not be loadable later on'
    )
    parser.add_argument(
        '-ssp',
        '--skip_save_progress',
        action='store_true',
        default=False,
        help='disables saving progress each epoch. By default Ludwig saves '
             'weights and stats after each epoch for enabling resuming '
             'of training, but if the model is really big that can be '
             'time consuming and will uses twice as much space, use '
             'this parameter to skip it, but training cannot be resumed '
             'later on'
    )
    parser.add_argument(
        '-ssl',
        '--skip_save_log',
        action='store_true',
        default=False,
        help='disables saving TensorBoard logs. By default Ludwig saves '
             'logs for the TensorBoard, but if it is not needed turning it off '
             'can slightly increase the overall speed'
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
        '-g',
        '--gpus',
        nargs='+',
        type=int,
        default=None,
        help='list of GPUs to use'
    )
    parser.add_argument(
        '-gml',
        '--gpu_memory_limit',
        type=int,
        default=None,
        help='maximum memory in MB to allocate per GPU device'
    )
    parser.add_argument(
        '-dpt',
        '--disable_parallel_threads',
        action='store_false',
        dest='allow_parallel_threads',
        help='disable TensorFlow from using multithreading for reproducibility'
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
        default=False,
        help='enables debugging mode'
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
    logger = logging.getLogger('ludwig.experiment')

    args.backend = initialize_backend(args.backend)
    if args.backend.is_coordinator():
        print_ludwig('Experiment', LUDWIG_VERSION)

    if args.k_fold is None:
        experiment_cli(**vars(args))
    else:
        kfold_cross_validate_cli(**vars(args))


if __name__ == '__main__':
    contrib_import()
    contrib_command("experiment", *sys.argv)
    cli(sys.argv[1:])
