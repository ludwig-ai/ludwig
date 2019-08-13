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

import tensorflow as tf

from ludwig.features.feature_registries import input_type_registry
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.misc import get_from_registry


logger = logging.getLogger(__name__)


def build_inputs(input_features,
                 regularizer,
                 dropout_rate,
                 is_training=True,
                 **kwargs):
    # ================ Inputs =============
    feature_representations = OrderedDict()
    input_features = topological_sort_feature_dependencies(input_features)
    for input_feature in input_features:
        feature_representation = build_single_input(input_feature,
                                                    regularizer,
                                                    dropout_rate,
                                                    is_training=is_training,
                                                    **kwargs)
        feature_representations[input_feature['name']] = feature_representation
    return feature_representations


def build_single_input(input_feature,
                       regularizer,
                       dropout_rate,
                       is_training=True,
                       **kwargs):
    scope_name = input_feature['name']
    logger.debug('- Input {} feature {}'.format(
        input_feature['type'],
        input_feature['name']
    ))
    if input_feature.get('tied_weights', None) is not None:
        scope_name = input_feature['tied_weights']

    with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
        input_feature_class = get_from_registry(
            input_feature['type'],
            input_type_registry
        )
        input_feature_obj = input_feature_class(input_feature)
        feature_representation = input_feature_obj.build_input(
            regularizer=regularizer,
            dropout_rate=dropout_rate, is_training=is_training,
            **kwargs)
        feature_representation['representation'] = tf.identity(
            feature_representation['representation'],
            name=scope_name)
    return feature_representation


dynamic_length_encoders = {
    'rnn',
    'embed'
}
