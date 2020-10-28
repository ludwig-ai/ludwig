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
import os
import json
import numpy as np
from typing import Union

from ludwig.constants import SEQUENCE, PREPROCESSING, NAME, HASH
from ludwig.constants import TEXT
from ludwig.constants import TIMESERIES
from ludwig.utils.misc_utils import hash_dict
from ludwig.utils.strings_utils import UNKNOWN_SYMBOL
from ludwig.utils.strings_utils import tokenizer_registry

SEQUENCE_TYPES = {SEQUENCE, TEXT, TIMESERIES}


def should_regularize(regularize_layers):
    regularize = False
    if isinstance(regularize_layers, bool) and regularize_layers:
        regularize = True
    elif (isinstance(regularize_layers, (list, tuple))
          and regularize_layers and regularize_layers[-1]):
        regularize = True
    return regularize


def set_str_to_idx(set_string, feature_dict, tokenizer_name):
    try:
        tokenizer = tokenizer_registry[tokenizer_name]()
    except ValueError:
        raise Exception('Tokenizer {} not supported'.format(tokenizer_name))

    out = [feature_dict.get(item, feature_dict[UNKNOWN_SYMBOL]) for item in
           tokenizer(set_string)]

    return np.array(out, dtype=np.int32)


def compute_feature_hash(feature: dict) -> str:
    preproc_hash = hash_dict(feature.get(PREPROCESSING, {}))
    return feature[NAME] + "_" + preproc_hash.decode('ascii')


def locate_feature_in_list(feature_list: list, feature_name: str) -> Union[dict, None]:
    # searchs a list of features for the feature with the specified name
    # returns either the requested feature dictionary or None
    for f in feature_list:
        if f[NAME] == feature_name:
            # return the requested feature dictionary
            return f

    # requested feature not found in the list
    return None

def retrieve_feature_hash(
    output_directory: str,
    feature_type: str, # 'input_features' or 'output_features'
    feature_name: str
) -> str:
    # retrieves feature hash for either input or output features from
    # the description.json file

    with open(os.path.join(output_directory, 'description.json'), 'r') as f:
        description_dict = json.load(f)
    feature_list = description_dict['config'][feature_type]

    feature = locate_feature_in_list(feature_list, feature_name)

    feature_hash = feature[HASH]

    return feature_hash
