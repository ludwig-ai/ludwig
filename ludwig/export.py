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

from ludwig.api import LudwigModel
from ludwig.contrib import contrib_command
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.neuropod_utils import \
    export_neuropod as utils_export_neuropod
from ludwig.utils.print_utils import logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def export_savedmodel(
        model_path: str,
        output_path: str = 'savedmodel',
        **kwargs
) -> None:
    """Exports a model to SavedModel

    # Inputs

    :param model_path: (str) filepath to pre-trained model.
    :param output_path: (str, default: `'savedmodel'`) directory to store the
        savedmodel

    # Return
    :returns: (`None`)
    """
    logger.info('Model path: {}'.format(model_path))
    logger.info('Output path: {}'.format(output_path))
    logger.info('\n')

    model = LudwigModel.load(model_path)
    os.makedirs(output_path, exist_ok=True)
    model.save_savedmodel(output_path)

    logger.info('Saved to: {0}'.format(output_path))


def export_neuropod(
        model_path,
        output_path='neuropod',
        model_name='neuropod',
        **kwargs
):
    """Exports a model to Neuropod

    # Inputs

    :param model_path: (str) filepath to pre-trained model.
    :param output_path: (str, default: `'neuropod'`)  directory to store the
        neuropod model.
    :param model_name: (str, default: `'neuropod'`) save neuropod under this
        name.

    # Return

    :returns: (`None`)
    """
    logger.info('Model path: {}'.format(model_path))
    logger.info('Output path: {}'.format(output_path))
    logger.info('\n')

    utils_export_neuropod(model_path, output_path, model_name)

    logger.info('Saved to: {0}'.format(output_path))


def cli_export_savedmodel(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and saves it as a SavedModel.',
        prog='ludwig export_savedmodel',
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

    # -----------------
    # Output parameters
    # -----------------
    parser.add_argument(
        '-od',
        '--output_path',
        type=str,
        help='path where to save the export model',
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

    args.logging_level = logging_level_registry[args.logging_level]
    logging.getLogger('ludwig').setLevel(
        args.logging_level
    )
    global logger
    logger = logging.getLogger('ludwig.export')

    print_ludwig('Export SavedModel', LUDWIG_VERSION)

    export_savedmodel(**vars(args))


def cli_export_neuropod(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and saves it as a Neuropod.',
        prog='ludwig export_neuropod',
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
        '-mn',
        '--model_name',
        help='model name',
        default='neuropod'
    )

    # -----------------
    # Output parameters
    # -----------------
    parser.add_argument(
        '-od',
        '--output_path',
        type=str,
        help='path where to save the export model',
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

    args.logging_level = logging_level_registry[args.logging_level]
    logging.getLogger('ludwig').setLevel(
        args.logging_level
    )
    global logger
    logger = logging.getLogger('ludwig.export')

    print_ludwig('Export Neuropod', LUDWIG_VERSION)

    export_neuropod(**vars(args))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'savedmodel':
            contrib_command("export_savedmodel", *sys.argv)
            cli_export_savedmodel(sys.argv[2:])
        elif sys.argv[1] == 'neuropod':
            contrib_command("export_neuropod", *sys.argv)
            cli_export_neuropod(sys.argv[2:])
        else:
            print('Unrecognized command')
    else:
        print('Unrecognized command')
