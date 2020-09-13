# !/usr/bin/env python
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
"""
    File name: LudwigModel.py
    Author: Piero Molino
    Date created: 5/21/2019
    Date last modified: 5/21/2019
    Python Version: 3+
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

import ludwig.contrib

ludwig.contrib.contrib_import()

import yaml

from ludwig.experiment import \
    kfold_cross_validate as experiment_kfold_cross_validate
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.print_utils import logging_level_registry

# TODO(refactor): consolidate NewLudwigModel and API
from ludwig.models.new_ludwig_model import NewLudwigModel as LudwigModel

logger = logging.getLogger(__name__)

# todo(refactor): this should be adapted to the new API / functionalities
def kfold_cross_validate(
        num_folds,
        model_definition=None,
        model_definition_file=None,
        data_csv=None,
        output_directory='results',
        random_seed=default_random_seed,
        **kwargs
):
    """Performs k-fold cross validation and returns result data structures.


    # Inputs
    
    :param num_folds: (int) number of folds to create for the cross-validation
    :param model_definition: (dict, default: None) a dictionary containing
           information needed to build a model. Refer to the
           [User Guide](http://ludwig.ai/user_guide/#model-definition)
           for details.
    :param model_definition_file: (string, optional, default: `None`) path to
           a YAML file containing the model definition. If available it will be
           used instead of the model_definition dict.
    :param data_csv: (dataframe, default: None)
    :param data_csv: (string, default: None)
    :param output_directory: (string, default: 'results')
    :param random_seed: (int) Random seed used k-fold splits.

    # Return

    :return: (tuple(kfold_cv_stats, kfold_split_indices), dict) a tuple of
            dictionaries `kfold_cv_stats`: contains metrics from cv run.
             `kfold_split_indices`: indices to split training data into
             training fold and test fold.
    """

    (kfold_cv_stats,
     kfold_split_indices) = experiment_kfold_cross_validate(
        num_folds,
        model_definition=model_definition,
        model_definition_file=model_definition_file,
        data_csv=data_csv,
        output_directory=output_directory,
        random_seed=random_seed
    )

    return kfold_cv_stats, kfold_split_indices


# todo(refactor): this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def test_train(
        data_csv,
        model_definition,
        batch_size=128,
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        debug=False,
        logging_level=logging.ERROR,
        **kwargs
):
    model = LudwigModel(
        model_definition=model_definition,
        logging_level=logging_level,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
    )

    train_stats, _ = model.train(
        dataset=data_csv,
        debug=debug,
    )

    logger.critical(train_stats)

    # predict
    predictions = model.predict(
        dataset=data_csv,
        batch_size=batch_size
    )

    logger.critical(predictions)


# todo(refactor): this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def test_train_online(
        data_csv,
        model_definition,
        batch_size=128,
        debug=False,
        logging_level=logging.ERROR,
        **kwargs
):
    # TODO(refactor)
    # model_definition = merge_with_defaults(model_definition)
    # data, train_set_metadata = build_dataset(
    #     data_csv,
    #     (model_definition['input_features'] +
    #      model_definition['output_features']),
    #     model_definition['preprocessing']
    # )
    #
    # ludwig_model = NewLudwigModel(model_definition, logging_level=logging_level)
    # ludwig_model.initialize_model(train_set_metadata=train_set_metadata)
    #
    # ludwig_model.train_online(
    #     data_csv=data_csv,
    #     batch_size=128,
    # )
    # ludwig_model.train_online(
    #     data_csv=data_csv,
    #     batch_size=128,
    # )
    #
    # # predict
    # predictions = ludwig_model.predict(
    #     data_csv=data_csv,
    #     batch_size=batch_size,
    # )
    # ludwig_model.close()
    # logger.critical(predictions)
    pass


# todo(refactor): this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def test_predict(
        data_csv,
        model_path,
        batch_size=128,
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        logging_level=logging.ERROR,
        **kwargs
):
    ludwig_model = LudwigModel.load(
        model_path,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        logging_level=logging_level)

    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
    )

    ludwig_model.close()
    logger.critical(predictions)

    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
    )

    logger.critical(predictions)


# todo(refactoring): this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def main(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script tests ludwig APIs.'
    )

    parser.add_argument(
        '-t',
        '--test',
        default='train',
        choices=['train', 'train_online', 'predict'],
        help='which test to run'
    )

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument('--data_csv', help='input data CSV file')
    parser.add_argument(
        '--train_set_metadata_json',
        help='input metadata JSON file'
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument('-m', '--model_path', help='model to load')
    parser.add_argument(
        '-md',
        '--model_definition',
        type=yaml.safe_load,
        help='model definition'
    )

    # ------------------
    # Generic parameters
    # ------------------
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=128,
        help='size of batches'
    )

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument(
        '-g',
        '--gpus',
        type=int,
        default=None,
        help='list of gpu to use'
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

    if args.test == 'train':
        test_train(**vars(args))
    elif args.test == 'train_online':
        test_train_online(**vars(args))
    elif args.test == 'predict':
        test_predict(**vars(args))
    else:
        logger.info('Unsupported test type')


if __name__ == '__main__':
    main(sys.argv[1:])
