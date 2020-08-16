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
import numpy as np

from ludwig.constants import SEQUENCE, TYPE
from ludwig.constants import TEXT
from ludwig.constants import TIMESERIES
from ludwig.features.feature_registries import input_type_registry, \
    output_type_registry
from ludwig.utils.misc_utils import get_from_registry
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


def update_model_definition_with_metadata(model_definition,
                                          training_set_metadata):
    # populate input features fields depending on data
    # model_definition = merge_with_defaults(model_definition)
    for input_feature in model_definition['input_features']:
        feature = get_from_registry(
            input_feature[TYPE],
            input_type_registry
        )
        feature.populate_defaults(input_feature)
        feature.update_model_definition_with_metadata(
            input_feature,
            training_set_metadata[input_feature['name']],
            model_definition=model_definition
        )

    # populate output features fields depending on data
    for output_feature in model_definition['output_features']:
        feature = get_from_registry(
            output_feature[TYPE],
            output_type_registry
        )
        feature.populate_defaults(output_feature)
        feature.update_model_definition_with_metadata(
            output_feature,
            training_set_metadata[output_feature['name']]
        )

    for feature in (
            model_definition['input_features'] +
            model_definition['output_features']
    ):
        if 'preprocessing' in feature:
            feature['preprocessing'] = training_set_metadata[feature['name']][
                'preprocessing'
            ]
