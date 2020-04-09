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
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LayerNormalization

logger = logging.getLogger(__name__)


class FCLayer(Layer):

    def __init__(
            self,
            fc_size=256,
            norm=None,
            activation='relu',
            dropout_rate=0,
    ):
        super(FCLayer, self).__init__()

        # TODO add initializer and regularizer
        self.layers = [Dense(fc_size)]

        if norm == 'batch':
            self.layers.append(BatchNormalization())
        elif norm == 'layer':
            self.layers.append(LayerNormalization())

        self.layers.append(Activation(activation))

        if dropout_rate > 0:
            self.layers.append(Dropout(dropout_rate))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden, training=training)

        return hidden


class FCStack(Layer):

    def __init__(
            self,
            layers=None,
            num_layers=1,
            default_fc_size=256,
            default_activation='relu',
            default_use_bias=True,
            default_norm=None,
            default_dropout_rate=0,
            default_weights_initializer='glorot_uniform',
            default_bias_initializer='zeros',
            default_weights_regularizer=None,
            default_bias_regularizer=None,
            # default_activity_regularizer=None,
            # default_weights_constraint=None,
            # default_bias_constraint=None,
            **kwargs
    ):
        super(FCStack, self).__init__()

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
            if 'dropout_rate' not in layer:
                layer['dropout_rate'] = default_dropout_rate
            if 'weights_initializer' not in layer:
                layer['weights_initializer'] = default_weights_initializer
            if 'weights_regularizer ' not in layer:
                layer['weights_regularizer '] = default_weights_regularizer

        self.stack = []

        for i, layer in enumerate(self.layers):
            with tf.variable_scope('fc_' + str(i)):
                self.stack.append(
                    FCLayer(
                        fc_size=layer['fc_size'],
                        norm=layer['norm'],
                        activation=layer['activation'],
                        dropout_rate=layer['dropout_rate']
                    )
                )

    def build(
            self,
            input_shape,
    ):
        super(FCStack, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        hidden = inputs
        for layer in self.stack:
            hidden = layer(hidden, training=training)
        return hidden

    def compute_output_shape(self, input_shape):
        if self.stack:
            return self.stack[-1].compute_output_shape(input_shape)
        return input_shape
