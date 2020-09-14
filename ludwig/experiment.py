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
import os
import sys

import yaml

from ludwig.api import LudwigModel, kfold_cross_validate
from ludwig.contrib import contrib_command, contrib_import
from ludwig.globals import LUDWIG_VERSION, set_on_master, is_on_master
from ludwig.utils.data_utils import save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)


def experiment_cli(
        model_definition,
        model_definition_file=None,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        data_format=None,
        training_set_metadata=None,
        experiment_name='experiment',
        model_name='run',
        model_load_path=None,
        model_resume_path=None,
        skip_save_training_description=False,
        skip_save_training_statistics=False,
        skip_save_model=False,
        skip_save_progress=False,
        skip_save_log=False,
        skip_save_processed_input=False,
        skip_save_unprocessed_output=False,
        skip_save_test_predictions=False,
        skip_save_test_statistics=False,
        skip_collect_predictions=False,
        skip_collect_overall_stats=False,
        skip_save_predictions=False,
        output_directory='results',
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=None,
        random_seed=default_random_seed,
        debug=False,
        logging_level=logging.INFO,
        **kwargs
):
    """Trains a model on a dataset's training and validation splits and
    uses it to predict on the test split.
    It saves the trained model and the statistics of training and testing.
    :param model_definition: Model definition which defines the different
           parameters of the model, features, preprocessing and training.
    :type model_definition: Dictionary
    :param model_definition_file: The file that specifies the model definition.
           It is a yaml file.
    :type model_definition_file: filepath (str)
    :param data_csv: A CSV file containing the input data which is used to
           train, validate and test a model. The CSV either contains a
           split column or will be split.
    :type data_csv: filepath (str)
    :param data_train_csv: A CSV file containing the input data which is used
           to train a model.
    :type data_train_csv: filepath (str)
    :param data_validation_csv: A CSV file containing the input data which is used
           to validate a model..
    :type data_validation_csv: filepath (str)
    :param data_test_csv: A CSV file containing the input data which is used
           to test a model.
    :type data_test_csv: filepath (str)
    :param data_hdf5: If the dataset is in the hdf5 format, this is used instead
           of the csv file.
    :type data_hdf5: filepath (str)
    :param data_train_hdf5: If the training set is in the hdf5 format, this is
           used instead of the csv file.
    :type data_train_hdf5: filepath (str)
    :param data_validation_hdf5: If the validation set is in the hdf5 format,
           this is used instead of the csv file.
    :type data_validation_hdf5: filepath (str)
    :param data_test_hdf5: If the test set is in the hdf5 format, this is
           used instead of the csv file.
    :type data_test_hdf5: filepath (str)
    :param training_set_metadata_json: If the dataset is in hdf5 format, this is
           the associated json file containing metadata.
    :type training_set_metadata_json: filepath (str)
    :param experiment_name: The name for the experiment.
    :type experiment_name: Str
    :param model_name: Name of the model that is being used.
    :type model_name: Str
    :param model_load_path: If this is specified the loaded model will be used
           as initialization (useful for transfer learning).
    :type model_load_path: filepath (str)
    :param model_resume_path: Resumes training of the model from the path
           specified. The difference with model_load_path is that also training
           statistics like the current epoch and the loss and performance so
           far are also resumed effectively continuing a previously interrupted
           training process.
    :type model_resume_path: filepath (str)
    :param skip_save_training_description: Disables saving
           the description JSON file.
    :type skip_save_training_description: Boolean
    :param skip_save_training_statistics: Disables saving
           training statistics JSON file.
    :type skip_save_training_statistics: Boolean
    :param skip_save_model: Disables
               saving model weights and hyperparameters each time the model
           improves. By default Ludwig saves model weights after each epoch
           the validation metric improves, but if the model is really big
           that can be time consuming if you do not want to keep
           the weights and just find out what performance can a model get
           with a set of hyperparameters, use this parameter to skip it,
           but the model will not be loadable later on.
    :type skip_save_model: Boolean
    :param skip_save_progress: Disables saving
           progress each epoch. By default Ludwig saves weights and stats
           after each epoch for enabling resuming of training, but if
           the model is really big that can be time consuming and will uses
           twice as much space, use this parameter to skip it, but training
           cannot be resumed later on.
    :type skip_save_progress: Boolean
    :param skip_save_log: Disables saving TensorBoard
           logs. By default Ludwig saves logs for the TensorBoard, but if it
           is not needed turning it off can slightly increase the
           overall speed..
    :type skip_save_log: Boolean
    :param skip_save_processed_input: If a CSV dataset is provided it is
           preprocessed and then saved as an hdf5 and json to avoid running
           the preprocessing again. If this parameter is False,
           the hdf5 and json file are not saved.
    :type skip_save_processed_input: Boolean
    :param skip_save_unprocessed_output: By default predictions and
           their probabilities are saved in both raw unprocessed numpy files
           containing tensors and as postprocessed CSV files
           (one for each output feature). If this parameter is True,
           only the CSV ones are saved and the numpy ones are skipped.
    :type skip_save_unprocessed_output: Boolean
    :param skip_save_test_predictions: skips saving test predictions CSV files
    :type skip_save_test_predictions: Boolean
    :param skip_save_test_statistics: skips saving test statistics JSON file
    :type skip_save_test_statistics: Boolean
    :param output_directory: The directory that will contain the training
           statistics, the saved model and the training progress files.
    :type output_directory: filepath (str)
    :param gpus: List of GPUs that are available for training.
    :type gpus: List
    :param gpu_memory_limit: maximum memory in MB to allocate per GPU device.
    :type gpu_memory_limit: Integer
    :param allow_parallel_threads: allow TensorFlow to use multithreading parallelism
           to improve performance at the cost of determinism.
    :type allow_parallel_threads: Boolean
    :param use_horovod: Flag for using horovod
    :type use_horovod: Boolean
    :param random_seed: Random seed used for weights initialization,
           splits and any other random function.
    :type random_seed: Integer
    :param debug: If true turns on tfdbg with inf_or_nan checks.
    :type debug: Boolean
    """
    set_on_master(use_horovod)

    if model_load_path:
        model = LudwigModel.load(model_load_path)
    else:
        model = LudwigModel(
            model_definition=model_definition,
            model_definition_fp=model_definition_file,
            logging_level=logging_level,
            use_horovod=use_horovod,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            random_seed=random_seed
        )
    (
        test_results,
        train_stats,
        preprocessed_data
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
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_test_predictions=skip_save_test_predictions,
        skip_save_test_statistics=skip_save_test_statistics,
        skip_collect_predictions=skip_collect_predictions,
        skip_collect_overall_stats=skip_collect_overall_stats,
        skip_save_predictions=skip_save_predictions,
        output_directory=output_directory,
        random_seed=random_seed,
        debug=debug,
    )

    return model, test_results, train_stats, preprocessed_data


def kfold_cross_validate_cli(
        k_fold,
        model_definition=None,
        model_definition_file=None,
        data_csv=None,
        output_directory='results',
        random_seed=default_random_seed,
        skip_save_k_fold_split_indices=False,
        **kwargs
):
    """Wrapper function to performs k-fold cross validation.

    # Inputs
    :param k_fold: (int) number of folds to create for the cross-validation
    :param model_definition: (dict, default: None) a dictionary containing
            information needed to build a model. Refer to the [User Guide]
           (http://ludwig.ai/user_guide/#model-definition) for details.
    :param model_definition_file: (string, optional, default: `None`) path to
           a YAML file containing the model definition. If available it will be
           used instead of the model_definition dict.
    :param data_csv: (string, default: None)
    :param output_directory: (string, default: 'results')
    :param random_seed: (int) Random seed used k-fold splits.
    :param skip_save_k_fold_split_indices: (boolean, default: False) Disables
            saving k-fold split indices

    :return: None
    """

    (kfold_cv_stats,
     kfold_split_indices) = kfold_cross_validate(
        k_fold,
        model_definition=model_definition,
        model_definition_file=model_definition_file,
        data_csv=data_csv,
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
        description='This script trains and tests a model',
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
        '--data_csv',
        help='input data CSV file. If it has a split column, it will be used '
             'for splitting (0: train, 1: validation, 2: test), otherwise the '
             'dataset will be randomly split'
    )
    parser.add_argument('--data_train_csv', help='input train data CSV file')
    parser.add_argument(
        '--data_validation_csv',
        help='input validation data CSV file'
    )
    parser.add_argument('--data_test_csv', help='input test data CSV file')

    parser.add_argument(
        '--data_hdf5',
        help='input data HDF5 file. It is an intermediate preprocess version of'
             ' the input CSV created the first time a CSV file is used in the '
             'same directory with the same name and a hdf5 extension'
    )
    parser.add_argument(
        '--data_train_hdf5',
        help='input train data HDF5 file. It is an intermediate preprocess '
             'version of the input CSV created the first time a CSV file is '
             'used in the same directory with the same name and a hdf5 '
             'extension'
    )
    parser.add_argument(
        '--data_validation_hdf5',
        help='input validation data HDF5 file. It is an intermediate preprocess'
             ' version of the input CSV created the first time a CSV file is '
             'used in the same directory with the same name and a hdf5 '
             'extension'
    )
    parser.add_argument(
        '--data_test_hdf5',
        help='input test data HDF5 file. It is an intermediate preprocess '
             'version of the input CSV created the first time a CSV file is '
             'used in the same directory with the same name and a hdf5 '
             'extension'
    )

    parser.add_argument(
        '--training_set_metadata_json',
        help='input metadata JSON file. It is an intermediate preprocess file'
             ' containing the mappings of the input CSV created the first time '
             'a CSV file is used in the same directory with the same name and a'
             ' json extension'
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
    model_definition = parser.add_mutually_exclusive_group(required=True)
    model_definition.add_argument(
        '-md',
        '--model_definition',
        type=yaml.safe_load,
        help='model definition'
    )
    model_definition.add_argument(
        '-mdf',
        '--model_definition_file',
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
        help='path of a the model directory to resume training of'
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
        '--skip_save_test_predictions',
        help='skips saving test predictions CSV files',
        action='store_true', default=False
    )
    parser.add_argument(
        '-sstes',
        '--skip_save_test_statistics',
        help='skips saving test statistics JSON file',
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
             'that can be time consuming if you do not want to keep '
             'the weights and just find out what performance can a model get '
             'with a set of hyperparameters, use this parameter to skip it,'
             'but the model will not be loadable later on'
    )
    parser.add_argument(
        '-ssp',
        '--skip_save_progress',
        action='store_true',
        default=False,
        help='disables saving progress each epoch. By default Ludwig saves '
             'weights and stats  after each epoch for enabling resuming '
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
        '-uh',
        '--use_horovod',
        action='store_true',
        default=None,
        help='uses horovod for distributed training'
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

    logging.getLogger('ludwig').setLevel(
        logging_level_registry[args.logging_level]
    )
    global logger
    logger = logging.getLogger('ludwig.experiment')

    set_on_master(args.use_horovod)

    if is_on_master():
        print_ludwig('Experiment', LUDWIG_VERSION)

    if args.k_fold is None:
        experiment_cli(**vars(args))
    else:
        kfold_cross_validate_cli(**vars(args))


if __name__ == '__main__':
    contrib_import()
    contrib_command("experiment", *sys.argv)
    cli(sys.argv[1:])
