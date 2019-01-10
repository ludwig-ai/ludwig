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
import copy
import os
import random
import subprocess
import sys
from collections import OrderedDict, Mapping

import numpy

import ludwig.globals


def get_experiment_description(model_definition,
                               dataset_type='generic',
                               data_csv=None,
                               data_hdf5=None,
                               metadata_json=None,
                               data_train_csv=None,
                               data_validation_csv=None,
                               data_test_csv=None,
                               data_train_hdf5=None,
                               data_validation_hdf5=None,
                               data_test_hdf5=None,
                               random_seed=None):
    is_a_git_repo = subprocess.call(['git', 'branch'],
                                    stderr=subprocess.STDOUT,
                                    stdout=open(os.devnull, 'w')) == 0

    description = OrderedDict()
    description['ludwig_version'] = ludwig.globals.LUDWIG_VERSION
    description['command'] = ' '.join(sys.argv)
    if is_a_git_repo:
        description['commit_hash'] = \
            subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(
                'utf-8')[:12]
    description['dataset_type'] = dataset_type
    if random_seed is not None:
        description['random_seed'] = random_seed
    if data_csv is not None:
        description['input_data'] = data_csv
    elif data_hdf5 is not None and metadata_json is not None:
        description['input_data'] = data_hdf5
        description['input_metadata'] = metadata_json
    elif data_train_csv is not None:
        description['input_data_train'] = data_train_csv
        if data_validation_csv is not None:
            description['input_data_validation'] = data_validation_csv
        if data_test_csv is not None:
            description['input_data_test'] = data_test_csv
    elif data_train_hdf5 is not None and metadata_json is not None:
        description['input_data_train'] = data_train_hdf5
        if data_validation_hdf5 is not None:
            description['input_data_validation'] = data_validation_hdf5
        if data_test_hdf5 is not None:
            description['input_data_test'] = data_test_hdf5
        description['input_metadata'] = metadata_json
    description['model_definition'] = model_definition

    return description


def set_random_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    numpy.random.seed(random_seed)


def merge_dict(dct, merge_dct):
    ''' Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    '''
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            dct[k] = merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError(
            'Key {} not supported, available options: {}'.format(
                key, registry.keys()
            )
        )


def set_default_value(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
