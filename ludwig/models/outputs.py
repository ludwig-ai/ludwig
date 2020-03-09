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

import tensorflow.compat.v1 as tf

from ludwig.features.feature_registries import output_type_registry
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.misc import get_from_registry

logger = logging.getLogger(__name__)


def build_outputs(
        output_features_def,
        hidden,
        regularizer,
        **kwargs
):
    output_features_def = topological_sort_feature_dependencies(output_features_def)
    output_features = {}

    for output_feature_def in output_features_def:
        output_feature = build_single_output(
            output_feature_def,
            hidden,
            output_features,
            regularizer,
            **kwargs
        )
        output_features[output_feature_def['name']] = output_feature

    return output_features


def build_single_output(
        output_feature_def,
        feature_hidden,
        other_output_features,
        regularizer,
        **kwargs
):
    logger.debug('- Output {} feature {}'.format(
        output_feature_def['type'],
        output_feature_def['name']
    ))
    with tf.variable_scope(output_feature_def['name']):
        feature_class = get_from_registry(
            output_feature_def['type'],
            output_type_registry
        )
        feature = feature_class(output_feature_def)
        weighted_train_mean_loss, weighted_eval_loss, output_tensors = feature.concat_dependencies_and_build_output(
            feature_hidden,
            other_output_features,
            regularizer,
            **kwargs
        )

    return weighted_train_mean_loss, weighted_eval_loss, output_tensors


def calculate_combined_loss(output_feature):
    output_train_losses.append(output_feature)
    output_eval_losses.append(of_eval_loss)
    output_tensors.update(of_output_tensors)

    train_combined_mean_loss = tf.reduce_sum(
        tf.stack(output_train_losses),
        name='train_combined_mean_loss')

    # todo re-add later
    # if regularizer is not None:
    #    regularization_losses = tf.get_collection(
    #        tf.GraphKeys.REGULARIZATION_LOSSES)
    #   if regularization_losses:
    #        regularization_loss = tf.add_n(regularization_losses)
    #        logger.debug('- Regularization losses: {0}'.format(
    #            regularization_losses))
    #
    #   else:
    #        regularization_loss = tf.constant(0.0)
    # else:
    regularization_loss = tf.constant(0.0)

    train_reg_mean_loss = tf.add(train_combined_mean_loss,
                                 regularization_loss,
                                 name='train_combined_regularized_mean_loss')

    eval_combined_loss = tf.reduce_sum(tf.stack(output_eval_losses),
                                       axis=0,
                                       name='eval_combined_loss')

    return train_reg_mean_loss, eval_combined_loss, regularization_loss, output_tensors
