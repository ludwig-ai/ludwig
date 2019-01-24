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
from pprint import pformat

import yaml

from ludwig.data.postprocessing import postprocess
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.globals import LUDWIG_VERSION, set_on_master, get_on_master
from ludwig.models.modules.measure_modules import get_best_function
from ludwig.predict import predict
from ludwig.predict import print_prediction_results
from ludwig.predict import save_prediction_outputs
from ludwig.predict import save_prediction_statistics
from ludwig.train import get_experiment_dir_name
from ludwig.train import get_file_names
from ludwig.train import train
from ludwig.train import update_model_definition_with_metadata
from ludwig.utils.data_utils import save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.defaults import merge_with_defaults
from ludwig.utils.misc import get_experiment_description
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_ludwig


def experiment(
        model_definition,
        model_definition_file=None,
        data_csv=None,
        data_train_csv=None,
        data_validation_csv=None,
        data_test_csv=None,
        data_hdf5=None,
        data_train_hdf5=None,
        data_validation_hdf5=None,
        data_test_hdf5=None,
        metadata_json=None,
        experiment_name='experiment',
        model_name='run',
        model_load_path=None,
        model_resume_path=None,
        skip_save_progress_weights=False,
        skip_save_processed_input=False,
        skip_save_unprocessed_output=False,
        output_directory='results',
        gpus=None,
        gpu_fraction=1.0,
        use_horovod=False,
        random_seed=default_random_seed,
        debug=False,
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
    :param data_csv: A CSV file contanining the input data which is used to
           train, validate and test a model. The CSV either contains a
           split column or will be split.
    :type data_csv: filepath (str)
    :param data_train_csv: A CSV file contanining the input data which is used
           to train a model.
    :type data_train_csv: filepath (str)
    :param data_validation_csv: A CSV file contanining the input data which is used
           to validate a model..
    :type data_validation_csv: filepath (str)
    :param data_test_csv: A CSV file contanining the input data which is used
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
    :param metadata_json: If the dataset is in hdf5 format, this is
           the associated json file containing metadata.
    :type metadata_json: filepath (str)
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
    :param skip_save_progress_weights: Skips saving the weights at the end of
           each epoch. If this is true, training cannot be resumed from the
           exactly the state at the end of the previous epoch.
    :type skip_save_progress_weights: Boolean
    :param skip_save_processed_input: If a CSV dataset is provided it is
           preprocessed and then saved as an hdf5 and json to avoid running
           the preprocessing again. If this parameter is False,
           the hdf5 and json file are not saved.
    :type skip_save_processed_input: Boolean
    :param skip_save_unprocessed_output: By default predictions and
           their probabilities are saved in both raw unprocessed numpy files
           contaning tensors and as postprocessed CSV files
           (one for each output feature). If this parameter is True,
           only the CSV ones are saved and the numpy ones are skipped.
    :type skip_save_unprocessed_output: Boolean
    :param output_directory: The directory that will contanin the training
           statistics, the saved model and the training procgress files.
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
    :type debug: Boolean
    """
    # set input features defaults
    if model_definition_file is not None:
        with open(model_definition_file, 'r') as def_file:
            model_definition = merge_with_defaults(yaml.load(def_file))
    else:
        model_definition = merge_with_defaults(model_definition)

    # setup directories and file names
    experiment_dir_name = None
    if model_resume_path is not None:
        if os.path.exists(model_resume_path):
            experiment_dir_name = model_resume_path
        else:
            if get_on_master():
                logging.info(
                    'Model resume path does not exists, '
                    'starting training from scratch'
                )
            model_resume_path = None
    if model_resume_path is None:
        experiment_dir_name = get_experiment_dir_name(
            output_directory,
            experiment_name,
            model_name,
            append_suffix=not use_horovod
        )
    description_fn, training_stats_fn, model_dir = get_file_names(
        experiment_dir_name
    )

    # save description
    description = get_experiment_description(
        model_definition,
        data_csv,
        data_train_csv,
        data_validation_csv,
        data_test_csv,
        data_hdf5,
        data_train_hdf5,
        data_validation_hdf5,
        data_test_hdf5,
        metadata_json,
        random_seed
    )
    save_json(description_fn, description)

    # print description
    if get_on_master():
        logging.info('Experiment name: {}'.format(experiment_name))
        logging.info('Model name: {}'.format(model_name))
        logging.info('Output path: {}'.format(experiment_dir_name))
        logging.info('')
        for key, value in description.items():
            logging.info('{}: {}'.format(key, pformat(value, indent=4)))
        logging.info('')

    # preprocess
    training_set, validation_set, test_set, metadata = preprocess_for_training(
        model_definition,
        data_csv=data_csv,
        data_train_csv=data_train_csv,
        data_validation_csv=data_validation_csv,
        data_test_csv=data_test_csv,
        data_hdf5=data_hdf5,
        data_train_hdf5=data_train_hdf5,
        data_validation_hdf5=data_validation_hdf5,
        data_test_hdf5=data_test_hdf5,
        metadata_json=metadata_json,
        skip_save_processed_input=skip_save_processed_input,
        preprocessing_params=model_definition[
            'preprocessing'],
        random_seed=random_seed)
    if get_on_master():
        logging.info('Training set: {0}'.format(training_set.size))
        logging.info('Validation set: {0}'.format(validation_set.size))
        logging.info('Test set: {0}'.format(test_set.size))

    # update model definition with metadata properties
    update_model_definition_with_metadata(model_definition, metadata)

    # run the experiment
    model, training_results = train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        model_definition=model_definition,
        save_path=model_dir,
        model_load_path=model_load_path,
        resume=model_resume_path is not None,
        skip_save_progress_weights=skip_save_progress_weights,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug
    )
    (
        train_trainset_stats,
        train_valisest_stats,
        train_testset_stats
    ) = training_results

    # grab the results of the model with highest validation test performance
    validation_field = model_definition['training']['validation_field']
    validation_measure = model_definition['training']['validation_measure']
    validation_field_result = train_valisest_stats[validation_field]

    # max or min depending on the measure
    best_function = get_best_function(validation_measure)
    epoch_best_vali_measure, best_vali_measure = best_function(
        enumerate(validation_field_result[validation_measure]),
        key=lambda pair: pair[1]
    )

    best_vali_measure_epoch_test_measure = train_testset_stats[
        validation_field
    ][validation_measure][epoch_best_vali_measure]

    # print the results of the model with highest validation test performance
    if get_on_master():
        logging.info('Best validation model epoch: {0}'.format(
            epoch_best_vali_measure + 1)
        )
        logging.info(
            'Best validation model {0} on validation set {1}: {2}'.format(
                validation_measure,
                validation_field,
                best_vali_measure)
        )
        logging.info('Best validation model {0} on test set {1}: {2}'.format(
            validation_measure,
            validation_field,
            best_vali_measure_epoch_test_measure)
        )

    # save training statistics
    save_json(
        training_stats_fn,
        {
            'train': train_trainset_stats,
            'validation': train_valisest_stats,
            'test': train_testset_stats
        }
    )

    # predict
    test_results = predict(
        test_set,
        model,
        model_definition,
        model_definition['training']['batch_size'],
        only_predictions=False,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        debug=debug
    )
    model.close_session()

    # postprocess
    postprocessed_output = postprocess(
        test_results,
        model_definition['output_features'],
        metadata,
        experiment_dir_name,
        skip_save_unprocessed_output
    )
    if get_on_master():
        print_prediction_results(test_results)

    save_prediction_outputs(postprocessed_output, experiment_dir_name)
    save_prediction_statistics(test_results, experiment_dir_name)

    if get_on_master():
        logging.info('\nFinished: {0}_{1}'.format(experiment_name, model_name))
        logging.info('Saved to: {}'.format(experiment_dir_name))


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script trains and tests a model.',
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
        '--metadata_json',
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

    # ----------------
    # Model parameters
    # ----------------
    model_definition = parser.add_mutually_exclusive_group(required=True)
    model_definition.add_argument(
        '-md',
        '--model_definition',
        type=yaml.load,
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
        '-sspw',
        '--skip_save_progress_weights',
        help='does not save weights after each epoch. By default Ludwig saves '
             'weights after each epoch for enabling resuming of training, but '
             'if the model is really big that can be time consuming and will '
             'use twice as much storage space, use this parameter to skip it.'
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

    logging.basicConfig(
        stream=sys.stdout,
        level=logging_level_registry[args.logging_level],
        format='%(message)s'
    )

    set_on_master(args.use_horovod)

    if get_on_master():
        print_ludwig('Experiment', LUDWIG_VERSION)

    experiment(**vars(args))


if __name__ == '__main__':
    cli(sys.argv[1:])
