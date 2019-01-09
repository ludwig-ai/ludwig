#! /usr/bin/env python
# coding=utf-8
# Copyright 2019 The Ludwig Authors. All Rights Reserved.
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

import numpy as np

from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import LUDWIG_VERSION
from ludwig.models.model import load_model_and_definition
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.print_utils import print_ludwig
from ludwig.utils.strings_utils import make_safe_filename


def collect_activations(
        model_path,
        tensors,
        data_csv=None,
        data_hdf5=None,
        metadata_json=None,
        dataset_type='generic',
        split='test',
        batch_size=128,
        output_directory='results',
        gpus=None,
        gpu_fraction=1.0,
        debug=False,
        **kwargs
):
    """Uses the pretrained model to collect the tensors corresponding to a
    datapoint in the dataset. Saves the tensors to the experiment directory

    :param model_path: Is the model from which the tensors will be collected
    :param tensors: The names of the tensors that are to be collected  # TODO: confirm this is true
    :param data_csv: The CSV filepath which contains the datapoints from which
           the tensors are collected
    :param data_hdf5: The HDF5 file path if the CSV file path does not exist,
           an alternative source of providing the data to the model
    :param dataset_type: Dataset type
    :param split: Split type
    :param batch_size: Batch size
    :param output_directory: Output directory
    :param metadata_json: Model metadata that is fed to the *preprocess*
           function
    :param gpus: The total number of GPUs that the model intends to use
    :param gpu_fraction: The fraction of each GPU that the model intends on
           using
    :param debug: To step through the stack traces and find possible errors
    :returns: None

    """
    # setup directories and file names
    experiment_dir_name = output_directory
    suffix = 0
    while os.path.exists(experiment_dir_name):
        experiment_dir_name = output_directory + '_' + str(suffix)
        suffix += 1

    logging.info('Dataset type: {}'.format(dataset_type))
    logging.info('Dataset path: {}'.format(
        data_csv if data_csv is not None else data_hdf5)
    )
    logging.info('Model path: {}'.format(model_path))
    logging.info('Output path: {}'.format(experiment_dir_name))
    logging.info('\n')

    # preprocessing
    dataset, metadata = preprocess_for_prediction(
        model_path,
        split,
        dataset_type,
        data_csv,
        data_hdf5,
        metadata_json
    )

    model, model_definition = load_model_and_definition(model_path)

    # collect activations
    print_boxed("COLLECT ACTIVATIONS")
    collected_tensors = model.collect_activations(
        dataset,
        tensors,
        batch_size,
        gpus=gpus,
        gpu_fraction=gpu_fraction
    )

    model.close_session()

    # saving
    os.mkdir(experiment_dir_name)
    save_tensors(collected_tensors, experiment_dir_name)

    logging.info('Saved to: {0}'.format(experiment_dir_name))


def collect_weights(
        model_path,
        tensors,
        output_directory='results',
        debug=False,
        **kwargs
):
    # setup directories and file names
    experiment_dir_name = output_directory
    suffix = 0
    while os.path.exists(experiment_dir_name):
        experiment_dir_name = output_directory + '_' + str(suffix)
        suffix += 1

    logging.info('Model path: {}'.format(model_path))
    logging.info('Output path: {}'.format(experiment_dir_name))
    logging.info('\n')

    model, model_definition = load_model_and_definition(model_path)

    # collect weights
    print_boxed("COLLECT WEIGHTS")
    collected_tensors = model.collect_weights(tensors)
    model.close_session()

    # saving
    os.mkdir(experiment_dir_name)
    save_tensors(collected_tensors, experiment_dir_name)

    logging.info('Saved to: {0}'.format(experiment_dir_name))


def save_tensors(collected_tensors, experiment_dir_name):
    for tensor_name, tensor_values in collected_tensors.items():
        np_filename = os.path.join(
            experiment_dir_name,
            make_safe_filename(tensor_name) + '.npy'
        )
        np.save(np_filename, tensor_values)


def cli_collect_activations(sys_argv):
    """Command Line Interface to communicate with the collection of tensors and
    there are several options that can specified when calling this function:

    --data_csv: Filepath for the input csv
    --data_hdf5: Filepath for the input hdf5 file, if there is a csv file, this
                 is not read
    --metadata_json: The metadata for the model definition  # TODO: Could be a little more specific
    --d: Refers to the dataset type of the file being read, by default is
         *generic*
    --s: Refers to the split of the data, can be one of: train, test,
         validation, full
    --m: Input model that is necessary to collect to the tensors, this is a
         required *option*
    --t: Tensors to collect
    --od: Output directory of the model, defaults to results
    --bs: Batch size
    --g: Number of gpus that are to be used
    --gf: Fraction of each GPUs memory to use.
    --dbg: Debug if the model is to be started with python debugger
    --v: Verbose: Defines the logging level that the user will be exposed to
    """
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model and uses it collect '
                    'tensors for each datapoint in the dataset.',
        prog='ludwig collect_activations',
        usage='%(prog)s [options]')

    # ---------------
    # Data parameters
    # ---------------
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_csv', help='input data CSV file')
    group.add_argument('--data_hdf5', help='input data HDF5 file')
    parser.add_argument('--metadata_json', help='input metadata JSON file')

    parser.add_argument(
        '-s',
        '--split',
        default='test',
        choices=['training', 'validation', 'test', 'full'],
        help='the split to test the model on'
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument(
        '-m',
        '--model_path',
        help='model to load',
        required=True
    )
    parser.add_argument(
        '-t',
        '--tensors',
        help='tensors to collect',
        nargs='+',
        required=True
    )

    # -------------------------
    # Output results parameters
    # -------------------------
    parser.add_argument(
        '-od',
        '--output_directory',
        type=str,
        default='results',
        help='directory that contains the results'
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
        default=0,
        help='list of gpu to use'
    )
    parser.add_argument(
        '-gf',
        '--gpu_fraction',
        type=float,
        default=1.0,
        help='fraction of gpu memory to initialize the process with'
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
        choices=["critical", "error", "warning", "info", "debug", "notset"]
    )

    args = parser.parse_args(sys_argv)

    logging.basicConfig(
        stream=sys.stdout,
        # filename='log.log', TODO - remove these?
        # filemode='w',
        level=logging_level_registry[args.logging_level],
        format='%(message)s'
    )

    print_ludwig('Collect Activations', LUDWIG_VERSION)

    collect_activations(**vars(args))


def cli_collect_weights(sys_argv):
    """Command Line Interface to collecting the weights for the model
    --m: Input model that is necessary to collect to the tensors, this is a
         required *option*
    --t: Tensors to collect
    --od: Output directory of the model, defaults to results
    --dbg: Debug if the model is to be started with python debugger
    --v: Verbose: Defines the logging level that the user will be exposed to
    """
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and uses it collect weights.',
        prog='ludwig collect_weights',
        usage='%(prog)s [options]'
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument(
        '-m',
        '--model_path',
        help='model to load',
        required=True
    )
    parser.add_argument(
        '-t',
        '--tensors',
        help='tensors to collect',
        nargs='+',
        required=True
    )

    # -------------------------
    # Output results parameters
    # -------------------------
    parser.add_argument(
        '-od',
        '--output_directory',
        type=str,
        default='results',
        help='directory that contains the results'
    )

    # ------------------
    # Runtime parameters
    # ------------------
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
        choices=["critical", "error", "warning", "info", "debug", "notset"]
    )

    args = parser.parse_args(sys_argv)

    logging.basicConfig(
        stream=sys.stdout,
        # filename='log.log', TODO - remove these?
        # filemode='w',
        level=logging_level_registry[args.logging_level],
        format='%(message)s'
    )

    print_ludwig('Collect Weights', LUDWIG_VERSION)
    collect_weights(**vars(args))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'activations':
            cli_collect_activations(sys.argv[2:])
        elif sys.argv[1] == 'weights':
            cli_collect_weights(sys.argv[2:])
        else:
            print('Unrecognized command')
    else:
        print('Unrecognized command')
