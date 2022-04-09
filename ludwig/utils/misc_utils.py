#! /usr/bin/env python
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
from collections import OrderedDict
from collections.abc import Mapping

import numpy
import torch

from ludwig.constants import PROC_COLUMN
from ludwig.utils.fs_utils import find_non_existing_dir_by_adding_suffix


def set_random_seed(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


def merge_dict(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of updating only top-level keys,
    dict_merge recurses down into dicts nested to an arbitrary depth, updating keys. The ``merge_dct`` is merged
    into ``dct``.

    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
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
                    summed_dict[key] = sum_dicts([prev_value, value], dict_type=type(value))
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
            key_in_dict2 = value[len(dict2_name) :]
            if key_in_dict2 in dict2.keys():
                value = dict2[key_in_dict2]
                resolved_dict[key] = value
    return resolved_dict


def get_from_registry(key, registry):
    if hasattr(key, "lower"):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key {key} not supported, available options: {registry.keys()}")


def set_default_value(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value


def set_default_values(dictionary, default_value_dictionary):
    # Set multiple default values
    for key, value in default_value_dictionary.items():
        set_default_value(dictionary, key, value)


def get_class_attributes(c):
    return {i for i in dir(c) if not callable(getattr(c, i)) and not i.startswith("_")}


def get_output_directory(output_directory, experiment_name, model_name="run"):
    base_dir_name = os.path.join(output_directory, experiment_name + ("_" if model_name else "") + (model_name or ""))
    return os.path.abspath(find_non_existing_dir_by_adding_suffix(base_dir_name))


def get_file_names(output_directory):
    description_fn = os.path.join(output_directory, "description.json")
    training_stats_fn = os.path.join(output_directory, "training_statistics.json")

    model_dir = os.path.join(output_directory, "model")

    return description_fn, training_stats_fn, model_dir


def get_combined_features(config):
    return config["input_features"] + config["output_features"]


def get_proc_features(config):
    return get_proc_features_from_lists(config["input_features"], config["output_features"])


def get_proc_features_from_lists(*args):
    return {feature[PROC_COLUMN]: feature for features in args for feature in features}
