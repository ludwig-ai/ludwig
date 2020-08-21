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

from ludwig.contrib import contrib_command, contrib_import
from ludwig.globals import LUDWIG_VERSION, is_on_master, set_on_master
from ludwig.models.new_ludwig_model import NewLudwigModel
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_ludwig

logger = logging.getLogger(__name__)


def predict_cli(
        model_path,
        dataset=None,
        data_format=None,
        batch_size=128,
        skip_save_unprocessed_output=False,
        skip_save_predictions=False,
        output_directory='results',
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=None,
        logging_level=logging.INFO,
        debug=False,
        **kwargs
):
    model = NewLudwigModel.load(
        model_path,
        logging_level=logging_level,
        use_horovod=use_horovod,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads
    )
    model.predict(
        dataset=dataset,
        data_format=data_format,
        batch_size=batch_size,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        output_directory=output_directory,
        return_type=dict,
        debug=debug,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and uses it to predict',
        prog='ludwig predict',
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
             'otherwise the dataset will be randomly split',
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
    parser.add_argument(
        '-ssuo',
        '--skip_save_unprocessed_output',
        help='skips saving intermediate NPY output files',
        action='store_true', default=False
    )
    parser.add_argument(
        '-sstp',
        '--skip_save_predictions',
        help='skips saving predictions CSV files',
        action='store_true', default=False
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
    logger = logging.getLogger('ludwig.predict')

    set_on_master(args.use_horovod)

    if is_on_master():
        print_ludwig('Predict', LUDWIG_VERSION)
        logger.info('Dataset path: {}'.format(args.dataset))
        logger.info('Model path: {}'.format(args.model_path))
        logger.info('')

    predict_cli(**vars(args))


if __name__ == '__main__':
    contrib_import()
    contrib_command("predict", *sys.argv)
    cli(sys.argv[1:])
