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
import base64
import copy
import hashlib
import json
import os
import random
import subprocess
import subprocess as sp
import sys
from collections import OrderedDict
from collections.abc import Mapping
from typing import Union

import numpy

import ludwig.globals
from ludwig.constants import PROC_COLUMN
from ludwig.utils.data_utils import figure_data_format


def get_experiment_description(
        config,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        data_format=None,
        random_seed=None
):
    description = OrderedDict()
    description['ludwig_version'] = ludwig.globals.LUDWIG_VERSION
    description['command'] = ' '.join(sys.argv)

    try:
        with open(os.devnull, 'w') as devnull:
            is_a_git_repo = subprocess.call(['git', 'branch'],
                                            stderr=subprocess.STDOUT,
                                            stdout=devnull) == 0
        if is_a_git_repo:
            description['commit_hash'] = \
                subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(
                    'utf-8')[:12]
    except:
        pass

    if random_seed is not None:
        description['random_seed'] = random_seed

    if isinstance(dataset, str):
        description['dataset'] = dataset
    if isinstance(training_set, str):
        description['training_set'] = training_set
    if isinstance(validation_set, str):
        description['validation_set'] = validation_set
    if isinstance(test_set, str):
        description['test_set'] = test_set
    if training_set_metadata is not None:
        description['training_set_metadata'] = training_set_metadata

    # determine data format if not provided or auto
    if not data_format or data_format == 'auto':
        data_format = figure_data_format(
            dataset, training_set, validation_set, test_set
        )

    if data_format:
        description['data_format'] = str(data_format)

    description['config'] = config

    import tensorflow as tf
    description['tf_version'] = tf.__version__

    return description


def set_random_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    numpy.random.seed(random_seed)


def merge_dict(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            dct[k] = merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def sum_dicts(dicts, dict_type=dict):
    summed_dict = dict_type()
    for d in dicts:
        for key, value in d.items():
            if key in summed_dict:
                prev_value = summed_dict[key]
                if isinstance(value, (dict, OrderedDict)):
                    summed_dict[key] = sum_dicts([prev_value, value],
                                                 dict_type=type(value))
                elif isinstance(value, numpy.ndarray):
                    summed_dict[key] = numpy.concatenate((prev_value, value))
                else:
                    summed_dict[key] = prev_value + value
            else:
                summed_dict[key] = value
    return summed_dict


def resolve_pointers(dict1, dict2, dict2_name):
    resolved_dict = copy.deepcopy(dict1)
    for key in dict1:
        value = dict1[key]
        if value.startswith(dict2_name):
            key_in_dict2 = value[len(dict2_name):]
            if key_in_dict2 in dict2.keys():
                value = dict2[key_in_dict2]
                resolved_dict[key] = value
    return resolved_dict


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


def set_default_values(dictionary, default_value_dictionary):
    # Set multiple default values
    for key, value in default_value_dictionary.items():
        set_default_value(dictionary, key, value)


def find_non_existing_dir_by_adding_suffix(directory_name):
    curr_directory_name = directory_name
    suffix = 0
    while os.path.exists(curr_directory_name):
        curr_directory_name = directory_name + '_' + str(suffix)
        suffix += 1
    return curr_directory_name


def get_class_attributes(c):
    return set(
        i for i in dir(c)
        if not callable(getattr(c, i)) and not i.startswith("_")
    )


def get_available_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[
                           1:]
        memory_free_values = [int(x.split()[0])
                              for i, x in enumerate(memory_free_info)]
    except Exception as e:
        print('"nvidia-smi" is probably not installed.', e)

    return memory_free_values


def get_output_directory(
        output_directory,
        experiment_name,
        model_name='run'
):
    base_dir_name = os.path.join(
        output_directory,
        experiment_name + ('_' if model_name else '') + model_name
    )
    return find_non_existing_dir_by_adding_suffix(base_dir_name)


def get_file_names(output_directory):
    description_fn = os.path.join(output_directory, 'description.json')
    training_stats_fn = os.path.join(
        output_directory, 'training_statistics.json')

    model_dir = os.path.join(output_directory, 'model')

    return description_fn, training_stats_fn, model_dir


def check_which_config(config, config_file):
    # check for config and config_file
    if config is None and config_file is None:
        raise ValueError(
            'Either config of config_file have to be'
            'not None to initialize a LudwigModel'
        )
    if config is not None and config_file is not None:
        raise ValueError(
            'Only one between config and '
            'config_file can be provided'
        )
    if not config:
        config = config_file
    return config


def hash_dict(d: dict, max_length: Union[int, None] = 6) -> bytes:
    s = json.dumps(d, sort_keys=True, ensure_ascii=True)
    h = hashlib.md5(s.encode())
    d = h.digest()
    b = base64.b64encode(d, altchars=b'__')
    return b[:max_length]


def get_combined_features(config):
    return config['input_features'] + config['output_features']


def get_proc_features(config):
    return get_proc_features_from_lists(config['input_features'], config['output_features'])


def get_proc_features_from_lists(*args):
    return {feature[PROC_COLUMN]: feature for features in args for feature in features}
