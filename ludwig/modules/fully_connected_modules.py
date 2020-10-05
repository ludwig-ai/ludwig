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

from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, Layer, LayerNormalization)

logger = logging.getLogger(__name__)


class FCLayer(Layer):

    def __init__(
            self,
            fc_size=256,
            use_bias=True,
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm=None,
            norm_params=None,
            activation='relu',
            dropout=0,
    ):
        super(FCLayer, self).__init__()

        self.layers = []

        self.layers.append(Dense(
            units=fc_size,
            use_bias=use_bias,
            kernel_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            # weights_constraint=weights_constraint,
            # bias_constraint=bias_constraint,
        ))

        if norm and norm_params is None:
            norm_params = {}
        if norm == 'batch':
            self.layers.append(BatchNormalization(**norm_params))
        elif norm == 'layer':
            self.layers.append(LayerNormalization(**norm_params))

        self.layers.append(Activation(activation))

        if dropout > 0:
            self.layers.append(Dropout(dropout))

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

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
            default_use_bias=True,
            default_weights_initializer='glorot_uniform',
            default_bias_initializer='zeros',
            default_weights_regularizer=None,
            default_bias_regularizer=None,
            default_activity_regularizer=None,
            # default_weights_constraint=None,
            # default_bias_constraint=None,
            default_norm=None,
            default_norm_params=None,
            default_activation='relu',
            default_dropout=0,
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
            if 'use_bias' not in layer:
                layer['use_bias'] = default_use_bias
            if 'weights_initializer' not in layer:
                layer['weights_initializer'] = default_weights_initializer
            if 'bias_initializer' not in layer:
                layer['bias_initializer'] = default_bias_initializer
            if 'weights_regularizer' not in layer:
                layer['weights_regularizer'] = default_weights_regularizer
            if 'bias_regularizer' not in layer:
                layer['bias_regularizer'] = default_bias_regularizer
            if 'activity_regularizer' not in layer:
                layer['activity_regularizer'] = default_activity_regularizer
            # if 'weights_constraint' not in layer:
            #     layer['weights_constraint'] = default_weights_constraint
            # if 'bias_constraint' not in layer:
            #     layer['bias_constraint'] = default_bias_constraint
            if 'norm' not in layer:
                layer['norm'] = default_norm
            if 'norm_params' not in layer:
                layer['norm_params'] = default_norm_params
            if 'activation' not in layer:
                layer['activation'] = default_activation
            if 'dropout' not in layer:
                layer['dropout'] = default_dropout

        self.stack = []

        for i, layer in enumerate(self.layers):
            self.stack.append(
                FCLayer(
                    fc_size=layer['fc_size'],
                    use_bias=layer['use_bias'],
                    weights_initializer=layer['weights_initializer'],
                    bias_initializer=layer['bias_initializer'],
                    weights_regularizer=layer['weights_regularizer'],
                    bias_regularizer=layer['bias_regularizer'],
                    activity_regularizer=layer['activity_regularizer'],
                    # weights_constraint=layer['weights_constraint'],
                    # bias_constraint=layer['bias_constraint'],
                    norm=layer['norm'],
                    norm_params=layer['norm_params'],
                    activation=layer['activation'],
                    dropout=layer['dropout'],
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
