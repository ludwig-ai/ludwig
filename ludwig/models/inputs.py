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

import tensorflow.compat.v1 as tf

from ludwig.features.feature_registries import input_type_registry
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.misc import get_from_registry

logger = logging.getLogger(__name__)


def build_inputs(
        input_features_def,
        regularizer,
        **kwargs
):
    # ================ Inputs =============
    input_features = OrderedDict()
    input_features_def = topological_sort_feature_dependencies(input_features_def)
    for input_feature_def in input_features_def:
        input_features[input_feature_def['name']] = build_single_input(
            input_feature_def,
            regularizer,
            **kwargs
        )
    return input_features


def build_single_input(input_feature_def,
                       regularizer,
                       **kwargs):
    scope_name = input_feature_def['name']
    logger.debug('- Input {} feature {}'.format(
        input_feature_def['type'],
        input_feature_def['name']
    ))
    if input_feature_def.get('tied_weights', None) is not None:
        scope_name = input_feature_def['tied_weights']

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        input_feature_class = get_from_registry(
            input_feature_def['type'],
            input_type_registry
        )
        input_feature_obj = input_feature_class(input_feature_def)
    return input_feature_obj


dynamic_length_encoders = {
    'rnn',
    'embed'
}
