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

from torch.nn import (Linear, LayerNorm, Module, Dropout)

from ludwig.utils.torch_utils import (LudwigModule, initializers, activations)

logger = logging.getLogger(__name__)



class FCLayer(LudwigModule):

    def __init__(
            self,
            input_size,
            output_size=256,
            use_bias=True,
            weights_initializer='xavier_uniform',
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
        super().__init__()

        self.layers = []

        '''
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
        '''
        fc = Linear(
            in_features=input_size,
            out_features=output_size,
            bias=use_bias
        )

        self.layers.append(fc)

        weights_initializer = initializers[weights_initializer]
        weights_initializer(fc.weight)

        bias_initializer = initializers[bias_initializer]
        bias_initializer(fc.bias)

        if weights_regularizer:
            self.add_loss(lambda: reg_loss(fc.weight, weights_regularizer))
        if bias_regularizer:
            self.add_loss(lambda: reg_loss(fc.bias, bias_regularizer))
        if activity_regularizer:
            # Handle in forward call
            self.add_loss(lambda: self.activation_loss)


        if norm and norm_params is None:
            norm_params = {}
        if norm == 'batch':
            #self.layers.append(BatchNormalization(**norm_params))
            # might need if statement for 1d vs 2d? like images
            if len(input_size) > 3:
                self.layers.append(BatchNorm1d(**norm_params))
            else:
                self.layers.append(BatchNorm2d(**norm_params))
        elif norm == 'layer':
            self.layers.append(LayerNorm(**norm_params))

        # Dict for activation objects in pytorch?
        #self.layers.append(Activations(activation))
        self.layers.append(activations[activation]())
        self.activation_index = len(self.layers) - 1

        if dropout > 0:
            self.layers.append(Dropout(dropout))

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

    def forward(self, inputs, training=None, mask=None):
        self.training = training
        batch_size = inputs.shape[0]
        hidden = inputs

        for i, layer in enumerate(self.layers):
            #hidden = layer(hidden, training=training)
            hidden = layer(hidden)
            if i == self.activation_index:
                self.activation_loss = reg_loss(hidden, self.activity_regularizer)/batch_size

        return hidden


class FCStack(LudwigModule):

    def __init__(
            self,
            first_layer_input_size,
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
        super().__init__()

        if layers is None:
            self.layers = []
            for i in range(num_layers):
                self.layers.append({})
        else:
            self.layers = layers

        if len(self.layers) > 0:
            self.layers[0]['input_size'] = first_layer_input_size
        for i, layer in enumerate(self.layers):
            if i != 0:
                layer['input_size'] = self.layers[i-1]['fc_size']
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
                    input_size=layer['input_size'],
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

    '''
    def build(
            self,
            input_shape,
    ):
        super().build(input_shape)
    '''

    def forward(self, inputs, training=None, mask=None):
        hidden = inputs
        for layer in self.stack:
            hidden = layer(hidden, training=training)
        return hidden
