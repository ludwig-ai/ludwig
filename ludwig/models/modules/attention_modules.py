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

import tensorflow as tf


logger = logging.getLogger(__name__)


def reduce_feed_forward_attention(current_inputs, hidden_size=256):
    with tf.compat.v1.variable_scope('reduce_ff_attention'):
        weights_1 = tf.compat.v1.get_variable('weights_1',
                                    [current_inputs.shape[-1], hidden_size])
        logger.debug('  att_weights_1: {}'.format(weights_1))
        biases_1 = tf.compat.v1.get_variable('biases_1', [hidden_size])
        logger.debug('  att_biases_1: {}'.format(biases_1))
        weights_2 = tf.compat.v1.get_variable('weights_2', [hidden_size, 1])
        logger.debug('  att_weights_2: {}'.format(weights_2))

        current_inputs_reshape = tf.reshape(current_inputs,
                                            [-1, current_inputs.shape[-1]])
        hidden = tf.tanh(
            tf.matmul(current_inputs_reshape, weights_1) + biases_1)
        logger.debug('  att_hidden: {}'.format(hidden))
        attention = tf.nn.softmax(tf.reshape(tf.matmul(hidden, weights_2),
                                             [-1, tf.shape(current_inputs)[1]]))
        logger.debug('  att_attention: {}'.format(attention))
        # attention [bs x seq]
        geated_inputs = tf.reduce_sum(
            tf.expand_dims(attention, -1) * current_inputs, 1)
        logger.debug('  att_geated_inputs: {}'.format(geated_inputs))
    return geated_inputs


def feed_forward_attention(current_inputs, feature_hidden_size,
                           hidden_size=256):
    with tf.compat.v1.variable_scope('ff_attention'):
        geated_inputs = reduce_feed_forward_attention(current_inputs,
                                                      hidden_size=hidden_size)

        # stacking inputs and attention vectors
        tiled_geated_inputs = tf.tile(tf.expand_dims(geated_inputs, 1),
                                      [1, tf.shape(current_inputs)[1], 1])
        logger.debug(
            '  att_tiled_geated_inputs: {}'.format(tiled_geated_inputs))
        outputs = tf.concat([current_inputs, tiled_geated_inputs],
                            axis=-1)  # [bs x s1 x 2*h]
        logger.debug('  att_outputs: {}'.format(outputs))
        # outputs = current_inputs + context # [bs x s1 x h]

    return outputs, feature_hidden_size * 2


def simple_memory_attention(current_inputs, context):
    assert current_inputs.shape[2] == context.shape[2]
    # calculating attention
    attention = tf.nn.softmax(
        tf.matmul(current_inputs, context, transpose_b=True))  # [bs x s1 x s2]
    logger.debug('  att_outputs: {}'.format(attention))

    # weighted_sum(attention, encoding_sequence_embedded)
    exp_ese = tf.expand_dims(context, 1)  # [bs x 1 x s2 x h]
    exp_att = tf.expand_dims(attention, -1)  # [bs x s1 x s2 x 1]
    weighted_sum = tf.multiply(exp_ese, exp_att)  # [bs x s1 x s2 x h]
    reduced_weighted_sum = tf.reduce_sum(weighted_sum, axis=2)  # [bs x s1 x h]
    logger.debug('  att_reduced_weighted_sum: {}'.format(reduced_weighted_sum))

    # stacking inputs and attention vectors
    outputs = tf.concat([current_inputs, reduced_weighted_sum],
                        axis=-1)  # [bs x s1 x 2*h]
    logger.debug('  att_outputs: {}'.format(outputs))

    return outputs, outputs.shape[-1]


def feed_forward_memory_attention(current_inputs, memory, hidden_size=256):
    seq_len = tf.shape(current_inputs)[1]
    mem_len = tf.shape(current_inputs)[1]
    seq_width = current_inputs.shape[2]
    mem_width = memory.shape[2]

    inputs_tile = tf.reshape(tf.tile(current_inputs, [1, 1, mem_len]),
                             [-1, seq_len, mem_len, seq_width])
    context_tile = tf.reshape(tf.tile(memory, [1, seq_len, 1]),
                              [-1, seq_len, mem_len, mem_width])
    concat_tile = tf.concat([inputs_tile, context_tile],
                            axis=-1)  # [bs, seq, seq, seq_w + ctx_w]
    logger.debug('  att_input_context_concat: {}'.format(concat_tile))

    with tf.compat.v1.variable_scope('reduce_contextual_ff_attention'):
        weights_1 = tf.compat.v1.get_variable('weights_1',
                                    [concat_tile.shape[-1], hidden_size])
        logger.debug('  att_weights_1: {}'.format(weights_1))
        biases_1 = tf.compat.v1.get_variable('biases_1', [hidden_size])
        logger.debug('  att_biases_1: {}'.format(biases_1))
        weights_2 = tf.compat.v1.get_variable('weights_2', [hidden_size, 1])
        logger.debug('  att_weights_2: {}'.format(weights_2))

        current_inputs_reshape = tf.reshape(concat_tile,
                                            [-1, concat_tile.shape[-1]])
        hidden = tf.tanh(
            tf.matmul(current_inputs_reshape, weights_1) + biases_1)
        logger.debug('  att_hidden: {}'.format(hidden))
        attention = tf.nn.softmax(
            tf.reshape(tf.matmul(hidden, weights_2), [-1, seq_len, mem_len]))
        logger.debug('  att_attention: {}'.format(attention))
        # attention [bs x seq]
        geated_inputs = tf.reduce_sum(
            tf.expand_dims(attention, -1) * inputs_tile, 2)
        logger.debug('  att_geated_inputs: {}'.format(geated_inputs))

    return geated_inputs, geated_inputs.shape.as_list()[-1]
