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

from ludwig.models.modules.initializer_modules import get_initializer


logger = logging.getLogger(__name__)


def fc_layer(inputs, in_count, out_count,
             activation='relu', norm=None,
             is_training=True, weights=None, biases=None,
             dropout=False, dropout_rate=None,
             initializer=None, regularizer=None):
    if weights is None:
        if initializer is not None:
            initializer_obj = get_initializer(initializer)
            weights = tf.compat.v1.get_variable(
                'weights',
                initializer=initializer_obj([in_count, out_count]),
                regularizer=regularizer
            )
        else:
            if activation == 'relu':
                initializer = get_initializer('he_uniform')
            elif activation == 'sigmoid' or activation == 'tanh':
                initializer = get_initializer('glorot_uniform')
            # if initializer is None, tensorFlow seems to be using
            # a glorot uniform initializer
            weights = tf.compat.v1.get_variable(
                'weights',
                [in_count, out_count],
                regularizer=regularizer,
                initializer=initializer
            )

    logger.debug('  fc_weights: {}'.format(weights))

    if biases is None:
        biases = tf.compat.v1.get_variable('biases', [out_count],
                                 initializer=tf.constant_initializer(0.01))
    logger.debug('  fc_biases: {}'.format(biases))

    hidden = tf.matmul(inputs, weights) + biases

    if norm is not None:
        if norm == 'batch':
            hidden = tf.contrib.layers.batch_norm(hidden,
                                                  is_training=is_training)
        elif norm == 'layer':
            hidden = tf.contrib.layers.layer_norm(hidden)

    if activation:
        hidden = getattr(tf.nn, activation)(hidden)

    if dropout and dropout_rate is not None:
        hidden = tf.layers.dropout(hidden, rate=dropout_rate,
                                   training=is_training)
        logger.debug('  fc_dropout: {}'.format(hidden))

    return hidden


class FCStack:

    def __init__(
            self,
            layers=None,
            num_layers=1,
            default_fc_size=256,
            default_activation='relu',
            default_norm=None,
            default_dropout=False,
            default_initializer=None,
            default_regularize=True
    ):
        if layers is None:
            self.layers = []
            for i in range(num_layers):
                self.layers.append({})
        else:
            self.layers = layers

        for layer in self.layers:
            if 'fc_size' not in layer:
                layer['fc_size'] = default_fc_size
            if 'activation' not in layer:
                layer['activation'] = default_activation
            if 'norm' not in layer:
                layer['norm'] = default_norm
            if 'dropout' not in layer:
                layer['dropout'] = default_dropout
            if 'regularize' not in layer:
                layer['regularize'] = default_regularize
            if 'initializer' not in layer:
                layer['initializer'] = default_initializer
        
    def __call__(
            self,
            inputs,
            inputs_size,
            regularizer=None,
            dropout_rate=None,
            is_training=False
    ):
        hidden = inputs
        for i, layer in enumerate(self.layers):
            with tf.compat.v1.variable_scope('fc_' + str(i)):
                hidden = fc_layer(
                    hidden,
                    inputs_size,
                    layer['fc_size'],
                    activation=layer['activation'],
                    norm=layer['norm'],
                    dropout=layer['dropout'],
                    dropout_rate=dropout_rate,
                    is_training=is_training,
                    initializer=layer['initializer'],
                    regularizer=regularizer if layer[
                        'regularize'] else None
                )
                logger.debug('  fc_{}: {}'.format(i, hidden))

            inputs_size = layer['fc_size']

        return hidden
