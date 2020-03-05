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
import os.path
import sys
from pprint import pformat

import yaml

from ludwig.constants import TRAINING, VALIDATION, TEST
from ludwig.contrib import contrib_command
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import input_type_registry
from ludwig.features.feature_registries import output_type_registry
from ludwig.globals import LUDWIG_VERSION, set_on_master, is_on_master
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.models.model import Model
from ludwig.models.model import load_model_and_definition
from ludwig.models.modules.measure_modules import get_best_function
from ludwig.utils.data_utils import save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.defaults import merge_with_defaults
from ludwig.utils.misc import get_experiment_description, \
    find_non_existing_dir_by_adding_suffix
from ludwig.utils.misc import get_from_registry
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)


def full_train(
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
        output_directory='results',
        should_close_session=True,
        gpus=None,
        gpu_fraction=1.0,
        use_horovod=False,
        random_seed=42,
        debug=False,
        **kwargs
):
    """*full_train* defines the entire training procedure used by Ludwig's
    internals. Requires most of the parameters that are taken into the model.
    Builds a full ludwig model and performs the training.
    :param data_test_df:
    :param data_df:
    :param data_train_df:
    :param data_validation_df:
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
    :param train_set_metadata_json: If the dataset is in hdf5 format, this is
           the associated json file containing metadata.
    :type train_set_metadata_json: filepath (str)
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
           far are also resumed effectively cotinuing a previously interrupted
           training process.
    :type model_resume_path: filepath (str)
    :param skip_save_training_description: Disables saving
           the description JSON file.
    :type skip_save_training_description: Boolean
    :param skip_save_training_statistics: Disables saving
           training statistics JSON file.
    :type skip_save_training_statistics: Boolean
    :param skip_save_model: Disables saving model weights
           and hyperparameters each time the model
           improves. By default Ludwig saves model weights after each epoch
           the validation measure improves, but if the model is really big
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
    :param skip_save_processed_input: If a CSV dataset is provided it is
           preprocessed and then saved as an hdf5 and json to avoid running
           the preprocessing again. If this parameter is False,
           the hdf5 and json file are not saved.
    :type skip_save_processed_input: Boolean
    :param skip_save_log: Disables saving TensorBoard
           logs. By default Ludwig saves logs for the TensorBoard, but if it
           is not needed turning it off can slightly increase the
           overall speed..
    :type skip_save_progress: Boolean
    :param output_directory: The directory that will contain the training
           statistics, the saved model and the training progress files.
    :type output_directory: filepath (str)
    :param gpus: List of GPUs that are available for training.
    :type gpus: List
    :param gpu_fraction: Fraction of the memory of each GPU to use at
           the beginning of the training. The memory may grow elastically.
    :type gpu_fraction: Integer
    :param random_seed: Random seed used for weights initialization,
           splits and any other random function.
    :type random_seed: Integer
    :param debug: If true turns on tfdbg with inf_or_nan checks.
    :type debug: Boolean
    :returns: None
    """
    # check for model_definition and model_definition_file
    if model_definition is None and model_definition_file is None:
        raise ValueError(
            'Either model_definition of model_definition_file have to be'
            'not None to initialize a LudwigModel'
        )
    if model_definition is not None and model_definition_file is not None:
        raise ValueError(
            'Only one between model_definition and '
            'model_definition_file can be provided'
        )

    # merge with default model definition to set defaults
    if model_definition_file is not None:
        with open(model_definition_file, 'r') as def_file:
            model_definition = merge_with_defaults(yaml.safe_load(def_file))
    else:
        model_definition = merge_with_defaults(model_definition)

    # setup directories and file names
    experiment_dir_name = None
    if model_resume_path is not None:
        if os.path.exists(model_resume_path):
            experiment_dir_name = model_resume_path
        else:
            if is_on_master():
                logger.info(
                    'Model resume path does not exists, '
                    'starting training from scratch'
                )
            model_resume_path = None

    if model_resume_path is None:
        if is_on_master():
            experiment_dir_name = get_experiment_dir_name(
                output_directory,
                experiment_name,
                model_name
            )
        else:
            experiment_dir_name = '.'

    # if model_load_path is not None, load its train_set_metadata
    if model_load_path is not None:
        train_set_metadata_json = os.path.join(
            model_load_path,
            TRAIN_SET_METADATA_FILE_NAME
        )

    # if we are skipping all saving,
    # there is no need to create a directory that will remain empty
    should_create_exp_dir = not (
            skip_save_training_description and
            skip_save_training_statistics and
            skip_save_model and
            skip_save_progress and
            skip_save_log and
            skip_save_processed_input
    )
    if is_on_master():
        if should_create_exp_dir:
            if not os.path.exists(experiment_dir_name):
                os.makedirs(experiment_dir_name)

    description_fn, training_stats_fn, model_dir = get_file_names(
        experiment_dir_name
    )

    # save description
    description = get_experiment_description(
        model_definition,
        data_csv=data_csv,
        data_train_csv=data_train_csv,
        data_validation_csv=data_validation_csv,
        data_test_csv=data_test_csv,
        data_hdf5=data_hdf5,
        data_train_hdf5=data_train_hdf5,
        data_validation_hdf5=data_validation_hdf5,
        data_test_hdf5=data_test_hdf5,
        metadata_json=train_set_metadata_json,
        random_seed=random_seed
    )
    if is_on_master():
        if not skip_save_training_description:
            save_json(description_fn, description)
        # print description
        logger.info('Experiment name: {}'.format(experiment_name))
        logger.info('Model name: {}'.format(model_name))
        logger.info('Output path: {}'.format(experiment_dir_name))
        logger.info('\n')
        for key, value in description.items():
            logger.info('{}: {}'.format(key, pformat(value, indent=4)))
        logger.info('\n')

    # preprocess
    preprocessed_data = preprocess_for_training(
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
        skip_save_processed_input=skip_save_processed_input,
        preprocessing_params=model_definition['preprocessing'],
        random_seed=random_seed
    )

    (training_set,
     validation_set,
     test_set,
     train_set_metadata) = preprocessed_data

    if is_on_master():
        logger.info('Training set: {0}'.format(training_set.size))
        if validation_set is not None:
            logger.info('Validation set: {0}'.format(validation_set.size))
        if test_set is not None:
            logger.info('Test set: {0}'.format(test_set.size))

    # update model definition with metadata properties
    update_model_definition_with_metadata(
        model_definition,
        train_set_metadata
    )

    if is_on_master():
        if not skip_save_model:
            # save train set metadata
            os.makedirs(model_dir, exist_ok=True)
            save_json(
                os.path.join(
                    model_dir,
                    TRAIN_SET_METADATA_FILE_NAME
                ),
                train_set_metadata
            )

    contrib_command("train_init", experiment_directory=experiment_dir_name,
                    experiment_name=experiment_name, model_name=model_name,
                    output_directory=output_directory,
                    resume=model_resume_path is not None)

    # run the experiment
    model, result = train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        model_definition=model_definition,
        save_path=model_dir,
        model_load_path=model_load_path,
        resume=model_resume_path is not None,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug
    )

    train_trainset_stats, train_valisest_stats, train_testset_stats = result
    train_stats = {
        TRAINING: train_trainset_stats,
        VALIDATION: train_valisest_stats,
        TEST: train_testset_stats
    }

    if should_close_session:
        model.close_session()

    # save training statistics
    if is_on_master():
        if not skip_save_training_statistics:
            save_json(training_stats_fn, train_stats)

    # grab the results of the model with highest validation test performance
    validation_field = model_definition[TRAINING]['validation_field']
    validation_measure = model_definition[TRAINING]['validation_measure']
    validation_field_result = train_valisest_stats[validation_field]

    best_function = get_best_function(validation_measure)
    # results of the model with highest validation test performance
    if is_on_master() and validation_set is not None:
        epoch_best_vali_measure, best_vali_measure = best_function(
            enumerate(validation_field_result[validation_measure]),
            key=lambda pair: pair[1]
        )
        logger.info(
            'Best validation model epoch: {0}'.format(
                epoch_best_vali_measure + 1)
        )
        logger.info(
            'Best validation model {0} on validation set {1}: {2}'.format(
                validation_measure, validation_field, best_vali_measure
            ))
        if test_set is not None:
            best_vali_measure_epoch_test_measure = train_testset_stats[
                validation_field][validation_measure][epoch_best_vali_measure]

            logger.info(
                'Best validation model {0} on test set {1}: {2}'.format(
                    validation_measure,
                    validation_field,
                    best_vali_measure_epoch_test_measure
                )
            )
        logger.info('\nFinished: {0}_{1}'.format(experiment_name, model_name))
        logger.info('Saved to: {0}'.format(experiment_dir_name))

    contrib_command("train_save", experiment_dir_name)

    return (
        model,
        preprocessed_data,
        experiment_dir_name,
        train_stats,
        model_definition
    )


def train(
        training_set,
        validation_set,
        test_set,
        model_definition,
        save_path='model',
        model_load_path=None,
        resume=False,
        skip_save_model=False,
        skip_save_progress=False,
        skip_save_log=False,
        gpus=None,
        gpu_fraction=1.0,
        use_horovod=False,
        random_seed=default_random_seed,
        debug=False
):
    """
    :param training_set: Dataset contaning training data
    :type training_set: Dataset
    :param validation_set: Dataset contaning validation data
    :type validation_set: Datasetk
    :param test_set: Dataset contaning test data.
    :type test_set: Dataset
    :param model_definition: Model definition which defines the different
           parameters of the model, features, preprocessing and training.
    :type model_definition: Dictionary
    :param save_path: The path to save the model to.
    :type save_path: filepath (str)
    :param model_load_path: If this is specified the loaded model will be used
           as initialization (useful for transfer learning).
    :type model_load_path: filepath (str)
    :param skip_save_model: Disables
               saving model weights and hyperparameters each time the model
           improves. By default Ludwig saves model weights after each epoch
           the validation measure imrpvoes, but if the model is really big
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
    :param gpus: List of GPUs that are available for training.
    :type gpus: List
    :param gpu_fraction: Fraction of the memory of each GPU to use at
           the beginning of the training. The memory may grow elastically.
    :type gpu_fraction: Integer
    :param random_seed: Random seed used for weights initialization,
           splits and any other random function.
    :type random_seed: Integer
    :param debug: If true turns on tfdbg with inf_or_nan checks.
    :type debug: Boolean
    :returns: None
    """
    if model_load_path is not None:
        # Load model
        if is_on_master():
            print_boxed('LOADING MODEL')
            logger.info('Loading model: {}\n'.format(model_load_path))
        model, _ = load_model_and_definition(model_load_path,
                                             use_horovod=use_horovod)
    else:
        # Build model
        if is_on_master():
            print_boxed('BUILDING MODEL', print_fun=logger.debug)

        model = Model(
            model_definition['input_features'],
            model_definition['output_features'],
            model_definition['combiner'],
            model_definition[TRAINING],
            model_definition['preprocessing'],
            use_horovod=use_horovod,
            random_seed=random_seed,
            debug=debug
        )

    contrib_command("train_model", model, model_definition, model_load_path)

    # Train model
    if is_on_master():
        print_boxed('TRAINING')
    return model, model.train(
        training_set,
        validation_set=validation_set,
        test_set=test_set,
        save_path=save_path,
        resume=resume,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        random_seed=random_seed,
        **model_definition[TRAINING]
    )


def update_model_definition_with_metadata(model_definition, train_set_metadata):
    # populate input features fields depending on data
    # model_definition = merge_with_defaults(model_definition)
    for input_feature in model_definition['input_features']:
        feature = get_from_registry(
            input_feature['type'],
            input_type_registry
        )
        feature.populate_defaults(input_feature)
        feature.update_model_definition_with_metadata(
            input_feature,
            train_set_metadata[input_feature['name']],
            model_definition=model_definition
        )

    # populate output features fields depending on data
    for output_feature in model_definition['output_features']:
        feature = get_from_registry(
            output_feature['type'],
            output_type_registry
        )
        feature.populate_defaults(output_feature)
        feature.update_model_definition_with_metadata(
            output_feature,
            train_set_metadata[output_feature['name']]
        )

    for feature in (
            model_definition['input_features'] +
            model_definition['output_features']
    ):
        if 'preprocessing' in feature:
            feature['preprocessing'] = train_set_metadata[feature['name']][
                'preprocessing'
            ]


def get_experiment_dir_name(
        output_directory,
        experiment_name,
        model_name='run'
):
    base_dir_name = os.path.join(
        output_directory,
        experiment_name + ('_' if model_name else '') + model_name
    )
    return find_non_existing_dir_by_adding_suffix(base_dir_name)


def get_file_names(experiment_dir_name):
    description_fn = os.path.join(experiment_dir_name, 'description.json')
    training_stats_fn = os.path.join(
        experiment_dir_name, 'training_statistics.json')

    model_dir = os.path.join(experiment_dir_name, 'model')

    return description_fn, training_stats_fn, model_dir


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script trains a model',
        prog='ludwig train',
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
        help='input data CSV file. '
             'If it has a split column, it will be used for splitting '
             '(0: train, 1: validation, 2: test), '
             'otherwise the dataset will be randomly split'
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
        '--train_set_metadata_json',
        help='input metadata JSON file. It is an intermediate preprocess file '
             'containing the mappings of the input CSV created the first time a'
             ' CSV file is used in the same directory with the same name and a '
             'json extension'
    )

    parser.add_argument(
        '-sspi',
        '--skip_save_processed_input',
        help='skips saving intermediate HDF5 and JSON files',
        action='store_true',
        default=False
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
        '-ssm',
        '--skip_save_model',
        action='store_true',
        default=False,
        help='disables saving weights each time the model imrpoves. '
             'By default Ludwig saves  weights after each epoch '
             'the validation measure imrpvoes, but  if the model is really big '
             'that can be time consuming if you do not want to keep '
             'the weights and just find out what performance can a model get '
             'with a set of hyperparameters, use this parameter to skip it'
    )
    parser.add_argument(
        '-ssp',
        '--skip_save_progress',
        action='store_true',
        default=False,
        help='disables saving weights after each epoch. By default ludwig saves '
             'weights after each epoch for enabling resuming of training, but '
             'if the model is really big that can be time consuming and will '
             'save twice as much space, use this parameter to skip it'
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
        help='list of gpus to use'
    )
    parser.add_argument(
        '-gf',
        '--gpu_fraction',
        type=float,
        default=1.0,
        help='fraction of gpu memory to initialize the process with'
    )
    parser.add_argument(
        '-uh',
        '--use_horovod',
        action='store_true',
        default=False,
        help='uses horovod for distributed training'
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

    logging.getLogger('ludwig').setLevel(
        logging_level_registry[args.logging_level]
    )
    set_on_master(args.use_horovod)

    if is_on_master():
        print_ludwig('Train', LUDWIG_VERSION)

    full_train(**vars(args))


if __name__ == '__main__':
    contrib_command("train", *sys.argv)
    cli(sys.argv[1:])
