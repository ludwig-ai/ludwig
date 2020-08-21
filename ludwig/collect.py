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

import numpy as np

from ludwig.contrib import contrib_command
from ludwig.globals import LUDWIG_VERSION
from ludwig.models.new_ludwig_model import NewLudwigModel
from ludwig.utils.misc_utils import find_non_existing_dir_by_adding_suffix
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.print_utils import print_ludwig
from ludwig.utils.strings_utils import make_safe_filename

logger = logging.getLogger(__name__)


def collect_activations(
        model_path,
        layers,
        dataset,
        data_format=None,
        batch_size=128,
        output_directory='results',
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=None,
        debug=False,
        **kwargs
):
    """Uses the pretrained model to collect the tensors corresponding to a
    datapoint in the dataset. Saves the tensors to the experiment directory

    :param model_path: Is the model from which the tensors will be collected
    :param layers: List of layer names we wish to collect the output from
    :param data_csv: The CSV filepath which contains the datapoints from which
           the tensors are collected
    :param data_hdf5: The HDF5 file path if the CSV file path does not exist,
           an alternative source of providing the data to the model
    :param split: Split type
    :param batch_size: Batch size
    :param output_directory: Output directory
    :param gpus: The total number of GPUs that the model intends to use
    :param gpu_memory_limit: (int: default: `None`) maximum memory in MB to allocate
           per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow TensorFlow to use
           multithreading parallelism to improve performance at the cost of
           determinism.
    :param debug: To step through the stack traces and find possible errors
    :returns: None

    """
    # setup directories and file names
    experiment_dir_name = find_non_existing_dir_by_adding_suffix(
        output_directory)

    logger.info('Dataset path: {}'.format(dataset)
                )
    logger.info('Model path: {}'.format(model_path))
    logger.info('Output path: {}'.format(experiment_dir_name))
    logger.info('\n')

    model = NewLudwigModel.load(
        model_path,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        use_horovod=use_horovod
    )

    # collect activations
    print_boxed('COLLECT ACTIVATIONS')
    collected_tensors = model.collect_activations(
        layers,
        dataset,
        data_format=data_format,
        batch_size=batch_size,
        debug=debug
    )

    # saving
    os.makedirs(experiment_dir_name)
    saved_filenames = save_tensors(collected_tensors, experiment_dir_name)

    logger.info('Saved to: {0}'.format(experiment_dir_name))
    return saved_filenames


def collect_weights(
        model_path,
        tensors,
        output_directory='results',
        debug=False,
        **kwargs
):
    # setup directories and file names
    experiment_dir_name = find_non_existing_dir_by_adding_suffix(
        output_directory)

    logger.info('Model path: {}'.format(model_path))
    logger.info('Output path: {}'.format(experiment_dir_name))
    logger.info('\n')

    model = NewLudwigModel.load(model_path)

    # collect weights
    print_boxed('COLLECT WEIGHTS')
    collected_tensors = model.collect_weights(tensors)

    # saving
    os.makedirs(experiment_dir_name)
    saved_filenames = save_tensors(collected_tensors, experiment_dir_name)

    logger.info('Saved to: {0}'.format(experiment_dir_name))
    return saved_filenames


def save_tensors(collected_tensors, experiment_dir_name):
    filenames = []
    for tensor_name, tensor_value in collected_tensors:
        np_filename = os.path.join(
            experiment_dir_name,
            make_safe_filename(tensor_name) + '.npy'
        )
        np.save(np_filename, tensor_value.numpy())
        filenames.append(np_filename)
    return filenames


def print_model_summary(
        model_path,
        **kwargs
):
    model = NewLudwigModel.load(model_path)
    collected_tensors = model.collect_weights()
    names = [name for name, w in collected_tensors]

    keras_model = model.model.get_connected_model()
    keras_model.summary()

    print('\nLayers:\n')
    for layer in keras_model.layers:
        print(layer.name)

    print('\nWeights:\n')
    for name in names:
        print(name)


def cli_collect_activations(sys_argv):
    """Command Line Interface to communicate with the collection of tensors and
    there are several options that can specified when calling this function:

    --data_csv: Filepath for the input csv
    --data_hdf5: Filepath for the input hdf5 file, if there is a csv file, this
                 is not read
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
    parser.add_argument(
        '--dataset',
        help='input data file path',
        required=True
    )
    parser.add_argument(
        '--data_format',
        help='format of the input data',
        default='auto',
        choices=['auto', 'csv', 'hdf5']
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
    logger = logging.getLogger('ludwig.collect')

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
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    args = parser.parse_args(sys_argv)

    logging.getLogger('ludwig').setLevel(
        logging_level_registry[args.logging_level]
    )
    global logger
    logger = logging.getLogger('ludwig.collect')

    print_ludwig('Collect Weights', LUDWIG_VERSION)

    collect_weights(**vars(args))


def cli_collect_summary(sys_argv):
    """Command Line Interface to collecting a summary of the model layers and weights.
    --m: Input model that is necessary to collect to the tensors, this is a
         required *option*
    --v: Verbose: Defines the logging level that the user will be exposed to
    """
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and uses it collect weight names.',
        prog='ludwig collect_summary',
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

    # ------------------
    # Runtime parameters
    # ------------------
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
    logger = logging.getLogger('ludwig.collect')

    print_ludwig('Collect Summary', LUDWIG_VERSION)

    print_model_summary(**vars(args))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'activations':
            contrib_command("collect_activations", *sys.argv)
            cli_collect_activations(sys.argv[2:])
        elif sys.argv[1] == 'weights':
            contrib_command("collect_weights", *sys.argv)
            cli_collect_weights(sys.argv[2:])
        elif sys.argv[1] == 'names':
            contrib_command("collect_summary", *sys.argv)
            cli_collect_summary(sys.argv[2:])
        else:
            print('Unrecognized command')
    else:
        print('Unrecognized command')
