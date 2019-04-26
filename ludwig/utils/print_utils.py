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
import logging
from collections import OrderedDict
from pprint import pformat

logging_level_registry = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'notset': logging.NOTSET
}


def print_ludwig(message, ludwig_version):
    logging.info('\n'.join([' _         _        _      ',
                            '| |_  _ __| |_ __ _(_)__ _ ',
                            '| | || / _` \\ V  V / / _` |',
                            '|_|\\_,_\\__,_|\\_/\\_/|_\\__, |',
                            '                     |___/ ',
                            'ludwig v{1} - {0}'.format(message, ludwig_version),
                            '']
                           ))


def print_boxed(text, print_fun=logging.info):
    box_width = len(text) + 2
    print_fun('')
    print_fun('╒{}╕'.format('═' * box_width))
    print_fun('│ {} │'.format(text.upper()))
    print_fun('╘{}╛'.format('═' * box_width))
    print_fun('')


def repr_ordered_dict(d: OrderedDict):
    return '{' + ',\n  '.join('{}: {}'.format(x, pformat(y, indent=4))
                              for x, y in d.items()) + '}'
