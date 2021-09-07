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
import numpy as np

import torch
from torch import nn
from ludwig.utils.torch_utils import get_activation, LudwigModule

logger = logging.getLogger(__name__)


class Conv1DLayer(LudwigModule):

    def __init__(
            self,
            in_channels=1,
            out_channels=256,
            sequence_size=None,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation=1,
            groups=1,
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
            pool_function='max',
            pool_size=2,
            pool_strides=None,
            pool_padding='valid',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequence_size = sequence_size
        self.kernel_size = kernel_size
        self.stride = strides
        if padding == 'same' and kernel_size is not None:
            self.padding = (self.kernel_size - 1) // 2
        else:
            self.padding = 0
        self.dilation = dilation
        self.groups = groups
        self.pool_size = pool_size
        if pool_strides is None:
            self.pool_strides = pool_size
        else:
            self.pool_strides = pool_strides
        if pool_padding == 'same' and pool_size is not None:
            self.pool_padding = (self.pool_size - 1) // 2
        else:
            self.pool_padding = 0


        self.layers = []

        self.layers.append(nn.Conv1d(
            # filters=num_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,),
            stride=(strides,),
            padding=padding,
            dilation=(dilation,),
            # use_bias=use_bias,
            # kernel_initializer=weights_initializer,
            # bias_initializer=bias_initializer,
            # kernel_regularizer=weights_regularizer,
            # bias_regularizer=bias_regularizer,
            # activity_regularizer=activity_regularizer,
            # kernel_constraint=None,
            # bias_constraint=None,
        ))

        if norm and norm_params is None:
            norm_params = {}
        if norm == 'batch':
            self.layers.append(
                nn.BatchNorm1d(
                    num_features=out_channels,
                    **norm_params)
            )
        elif norm == 'layer':
            # todo(jmt): confirm the shape (N, C, L) or (N, L, C)
            # following code sequence based on this posting:
            #   https://discuss.pytorch.org/t/how-to-use-layer-norm-after-con-1d-layer/65284/9
            # convert from (N, C, L) -> (N, L, C) for layer norm
            self.layers.append(
                lambda x: x.transpose(1, 2)
            )
            self.layers.append(nn.LayerNorm(
                normalized_shape=out_channels,
                **norm_params)
            )
            # convert from (N, L, C) -> (N, C, L) for remainder of processing
            self.layers.append(
                lambda x: x.transpose(1, 2)
            )

        self.layers.append(get_activation(activation))

        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        if pool_size is not None:
            pool = nn.MaxPool1d
            if pool_function in {'average', 'avg', 'mean'}:
                pool = nn.AvgPool1d
            self.layers.append(pool(
                kernel_size=self.pool_size,
                stride=self.pool_strides,
                padding=self.pool_padding
            ))


        # todo: determine how to handle layer.name
        # for layer in self.layers:
        #     logger.debug('   {}'.format(layer.name))

    @property
    def input_shape(self):
        """ Returns the size of the input tensor without the batch dimension. """
        return (torch.Size([self.sequence_size, self.in_channels]))

    def forward(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_size, in_channels]
        # in Torch nomenclature (N, L, C)
        hidden = inputs
        # put in form compatible with Torch
        hidden = hidden.transpose(1, 2)

        for i, layer in enumerate(self.layers):
            # todo: determine how to handle training parameter in this call
            #       commented out to avoid unexpected parameter error
            hidden = layer(hidden)  # , training=training)

        # revert back to normal form
        hidden = hidden.transpose(1, 2)

        return hidden  # (batch_size, out_channels, seq_size)


class Conv1DStack(LudwigModule):

    def __init__(
            self,
            in_channels=1,
            max_sequence_length=None,
            layers=None,
            num_layers=None,
            default_num_filters=256,
            default_filter_size=3,
            default_strides=1,
            default_padding='same',
            default_dilation_rate=1,
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
            default_pool_function='max',
            default_pool_size=2,
            default_pool_strides=None,
            default_pool_padding=0,
            **kwargs
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.in_channels = in_channels

        if layers is None:
            if num_layers is None:
                self.layers = [
                    {'filter_size': 7, 'pool_size': 3, 'regularize': False},
                    {'filter_size': 7, 'pool_size': 3, 'regularize': False},
                    {'filter_size': 3, 'pool_size': None, 'regularize': False},
                    {'filter_size': 3, 'pool_size': None, 'regularize': False},
                    {'filter_size': 3, 'pool_size': None, 'regularize': True},
                    {'filter_size': 3, 'pool_size': 3, 'regularize': True}
                ]
            else:
                self.layers = []
                for i in range(num_layers):
                    self.layers.append({
                        'filter_size': default_filter_size,
                        'num_filters': default_num_filters,
                        'pool_size': default_pool_size,
                        'pool_strides': default_pool_strides}
                    )
        else:
            self.layers = layers

        for layer in self.layers:
            if 'num_filters' not in layer:
                layer['num_filters'] = default_num_filters
            if 'filter_size' not in layer:
                layer['filter_size'] = default_filter_size
            if 'strides' not in layer:
                layer['strides'] = default_strides
            if 'padding' not in layer:
                layer['padding'] = default_padding
            if 'dilation_rate' not in layer:
                layer['dilation_rate'] = default_dilation_rate
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
            if 'pool_function' not in layer:
                layer['pool_function'] = default_pool_function
            if 'pool_size' not in layer:
                layer['pool_size'] = default_pool_size
            if 'pool_strides' not in layer:
                layer['pool_strides'] = default_pool_strides
            if 'pool_padding' not in layer:
                layer['pool_padding'] = default_pool_padding

        self.stack = []

        prior_layer_channels = in_channels
        l_in = self.max_sequence_length  # torch L_in
        for i, layer in enumerate(self.layers):
            logger.debug('   stack layer {}'.format(i))
            self.stack.append(
                Conv1DLayer(
                    in_channels=prior_layer_channels,
                    out_channels=layer['num_filters'],
                    sequence_size=l_in,
                    kernel_size=layer['filter_size'],
                    strides=layer['strides'],
                    padding=layer['padding'],
                    dilation=layer['dilation_rate'],
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
                    pool_function=layer['pool_function'],
                    pool_size=layer['pool_size'],
                    pool_strides=layer['pool_strides'],
                    pool_padding=layer['pool_padding'],
                )
            )

            # retrieve number of channels from prior layer
            input_shape = self.stack[i].input_shape
            output_shape = self.stack[i].output_shape

            logger.debug(
                f'{self.__class__.__name__}: '
                f'input_shape {input_shape}, output shape {output_shape}'
            )

            # pass along shape for the input to the next layer
            l_in, prior_layer_channels = output_shape

    @property
    def input_shape(self):
        """ Returns the size of the input tensor without the batch dimension. """
        return (torch.Size([self.max_sequence_length, self.in_channels]))

    def forward(self, inputs, training=None, mask=None):
        hidden = inputs

        # todo: enumerate for debugging, remove after testing
        for i, layer in enumerate(self.stack):
            hidden = layer(hidden, training=training)

        if hidden.shape[1] == 0:
            raise ValueError(
                'The output of the conv stack has the second dimension '
                '(length of the sequence) equal to 0. '
                'This means that the combination of filter_size, padding, '
                'stride, pool_size, pool_padding and pool_stride is reduces '
                'the sequence length more than is possible. '
                'Try using "same" padding and reducing or eliminating stride '
                'and pool.'
            )

        return hidden


class ParallelConv1D(LudwigModule):

    def __init__(
            self,
            in_channels=1,
            max_sequence_length=None,
            layers=None,
            default_num_filters=256,
            default_filter_size=3,
            default_strides=1,
            default_padding='same',
            default_dilation_rate=1,
            default_use_bias=True,
            default_weights_initializer='xavier_uniform',
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
            default_pool_function='max',
            default_pool_size=None,
            default_pool_strides=None,
            default_pool_padding='valid',
            **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.max_sequence_length = max_sequence_length

        if layers is None:
            self.layers = [
                {'filter_size': 2},
                {'filter_size': 3},
                {'filter_size': 4},
                {'filter_size': 5}
            ]
        else:
            self.layers = layers

        for layer in self.layers:
            if 'num_filters' not in layer:
                layer['num_filters'] = default_num_filters
            if 'filter_size' not in layer:
                layer['filter_size'] = default_filter_size
            if 'strides' not in layer:
                layer['strides'] = default_strides
            if 'padding' not in layer:
                layer['padding'] = default_padding
            if 'dilation_rate' not in layer:
                layer['dilation_rate'] = default_dilation_rate
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
            if 'pool_function' not in layer:
                layer['pool_function'] = default_pool_function
            if 'pool_size' not in layer:
                layer['pool_size'] = default_pool_size
            if 'pool_strides' not in layer:
                layer['pool_strides'] = default_pool_strides
            if 'pool_padding' not in layer:
                layer['pool_padding'] = default_pool_padding

        self.parallel_layers = []

        for i, layer in enumerate(self.layers):
            logger.debug('   parallel layer {}'.format(i))
            self.parallel_layers.append(
                Conv1DLayer(
                    in_channels=self.in_channels,
                    out_channels=layer['num_filters'],
                    sequence_size=self.max_sequence_length,
                    kernel_size=layer['filter_size'],
                    strides=layer['strides'],
                    padding=layer['padding'],
                    dilation=layer['dilation_rate'],
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
                    pool_function=layer['pool_function'],
                    pool_size=layer['pool_size'],
                    pool_strides=layer['pool_strides'],
                    pool_padding=layer['pool_padding'],
                )
            )

            logger.debug(f'{self.__class__.__name__} layer {i}, input shape '
                         f'{self.parallel_layers[i].input_shape}, output shape '
                         f'{self.parallel_layers[i].output_shape}')

    @property
    def input_shape(self) -> torch.Size:
        """ Returns the size of the input tensor without the batch dimension. """
        return torch.Size([self.max_sequence_length, self.in_channels])

    def forward(self, inputs, training=None, mask=None):
        # inputs: [batch_size, in_channels, seq_size)

        hidden = inputs
        hiddens = []

        for layer in self.parallel_layers:
            hiddens.append(layer(hidden, training=training))
        hidden = torch.cat(hiddens, 2)

        if hidden.shape[1] == 0:
            raise ValueError(
                'The output of the conv stack has the second dimension '
                '(length of the sequence) equal to 0. '
                'This means that the combination of filter_size, padding, '
                'stride, pool_size, pool_padding and pool_stride reduces '
                'the sequence length more than is possible. '
                'Try using "same" padding and reducing or eliminating stride '
                'and pool.'
            )

        return hidden  # (batch_size, len(parallel_layers) * out_channels, seq_size)


class ParallelConv1DStack(LudwigModule):

    def __init__(
            self,
            in_channels=None,
            stacked_layers=None,
            max_sequence_length=None,
            default_num_filters=64,
            default_filter_size=3,
            default_strides=1,
            default_padding='same',
            default_dilation_rate=1,
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
            default_pool_function='max',
            default_pool_size=None,
            default_pool_strides=None,
            default_pool_padding='valid',
            **kwargs
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.in_channels = in_channels

        if stacked_layers is None:
            self.stacked_parallel_layers = [
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ],
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ],
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ]
            ]

        else:
            self.stacked_parallel_layers = stacked_layers

        for i, parallel_layers in enumerate(self.stacked_parallel_layers):
            for j in range(len(parallel_layers)):
                layer = parallel_layers[j]
                if 'num_filters' not in layer:
                    layer['num_filters'] = default_num_filters
                if 'filter_size' not in layer:
                    layer['filter_size'] = default_filter_size
                if 'strides' not in layer:
                    layer['strides'] = default_strides
                if 'padding' not in layer:
                    layer['padding'] = default_padding
                if 'dilation_rate' not in layer:
                    layer['dilation_rate'] = default_dilation_rate
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
                    layer[
                        'activity_regularizer'] = default_activity_regularizer
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
                if 'pool_function' not in layer:
                    layer['pool_function'] = default_pool_function
                if 'pool_size' not in layer:
                    if i == len(self.stacked_parallel_layers) - 1:
                        layer['pool_size'] = default_pool_size
                    else:
                        layer['pool_size'] = None
                if 'pool_strides' not in layer:
                    layer['pool_strides'] = default_pool_strides
                if 'pool_padding' not in layer:
                    layer['pool_padding'] = default_pool_padding

        self.stack = []
        num_channels = self.in_channels
        sequence_length = self.max_sequence_length
        for i, parallel_layers in enumerate(self.stacked_parallel_layers):
            logger.debug('   stack layer {}'.format(i))
            self.stack.append(ParallelConv1D(
                num_channels,
                sequence_length,
                layers=parallel_layers)
            )

            logger.debug(f'{self.__class__.__name__} layer {i}, input shape '
                         f'{self.stack[i].input_shape}, output shape '
                         f'{self.stack[i].output_shape}')

            # set input specification for the layer
            num_channels = self.stack[i].output_shape[1]
            sequence_length = self.stack[i].output_shape[0]

    @property
    def input_shape(self):
        """ Returns the size of the input tensor without the batch dimension. """
        return torch.Size([self.in_channels, self.max_sequence_length])

    def forward(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.stack:
            hidden = layer(hidden, training=training)

        if hidden.shape[2] == 0:
            raise ValueError(
                'The output of the conv stack has the second dimension '
                '(length of the sequence) equal to 0. '
                'This means that the compination of filter_size, padding, '
                'stride, pool_size, pool_padding and pool_stride is reduces '
                'the sequence length more than is possible. '
                'Try using "same" padding and reducing or eliminating stride '
                'and pool.'
            )

        return hidden


class Conv2DLayer(LudwigModule):

    def __init__(
            self,
            num_filters=256,
            filter_size=3,
            strides=(1, 1),
            padding='valid',
            dilation_rate=(1, 1),
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
            pool_function='max',
            pool_size=(2, 2),
            pool_strides=None,
            pool_padding='valid',
    ):
        super().__init__()

        self.layers = []

        self.layers.append(Conv2D(
            filters=num_filters,
            kernel_size=filter_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            # weights_constraint=None,
            # bias_constraint=None,
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

        if pool_size is not None:
            pool = MaxPool2D
            if pool_function in {'average', 'avg', 'mean'}:
                pool = AveragePooling2D
            self.layers.append(pool(
                pool_size=pool_size, strides=pool_strides, padding=pool_padding
            ))

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden, training=training)

        return hidden


class Conv2DStack(LudwigModule):

    def __init__(
            self,
            layers=None,
            num_layers=None,
            default_num_filters=256,
            default_filter_size=3,
            default_strides=(1, 1),
            default_padding='valid',
            default_dilation_rate=(1, 1),
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
            default_pool_function='max',
            default_pool_size=(2, 2),
            default_pool_strides=None,
            default_pool_padding='valid',
    ):
        super().__init__()

        if layers is None:
            if num_layers is None:
                self.layers = [
                    {'num_filters': 32},
                    {'num_filters': 64},
                ]
            else:
                self.layers = []
                for i in range(num_layers):
                    self.layers.append({
                        'filter_size': default_filter_size,
                        'num_filters': default_num_filters,
                        'pool_size': default_pool_size}
                    )
        else:
            self.layers = layers

        for layer in self.layers:
            if 'num_filters' not in layer:
                layer['num_filters'] = default_num_filters
            if 'filter_size' not in layer:
                layer['filter_size'] = default_filter_size
            if 'strides' not in layer:
                layer['strides'] = default_strides
            if 'padding' not in layer:
                layer['padding'] = default_padding
            if 'dilation_rate' not in layer:
                layer['dilation_rate'] = default_dilation_rate
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
            if 'pool_function' not in layer:
                layer['pool_function'] = default_pool_function
            if 'pool_size' not in layer:
                layer['pool_size'] = default_pool_size
            if 'pool_strides' not in layer:
                layer['pool_strides'] = default_pool_strides
            if 'pool_padding' not in layer:
                layer['pool_padding'] = default_pool_padding

        self.stack = []

        for i, layer in enumerate(self.layers):
            logger.debug('   stack layer {}'.format(i))
            self.stack.append(
                Conv2DLayer(
                    num_filters=layer['num_filters'],
                    filter_size=layer['filter_size'],
                    strides=layer['strides'],
                    padding=layer['padding'],
                    dilation_rate=layer['dilation_rate'],
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
                    pool_function=layer['pool_function'],
                    pool_size=layer['pool_size'],
                    pool_strides=layer['pool_strides'],
                    pool_padding=layer['pool_padding'],
                )
            )

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.stack:
            hidden = layer(hidden, training=training)

        return hidden


class Conv2DLayerFixedPadding(LudwigModule):

    def __init__(
            self,
            num_filters=256,
            filter_size=3,
            strides=1,
            weights_regularizer=None
    ):
        super().__init__()

        self.layers = []

        if strides > 1:
            self.layers.append(ZeroPadding2D(padding=((filter_size - 1) // 2)))

        self.layers.append(
            Conv2D(
                filters=num_filters,
                kernel_size=filter_size,
                strides=strides,
                padding=('SAME' if strides == 1 else 'VALID'),
                use_bias=False,
                kernel_initializer=VarianceScaling(),
                # kernel_regularizer=weights_regularizer,
            )
        )

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden, training=training)

        return hidden


class ResNetBlock(LudwigModule):

    def __init__(
            self,
            num_filters,
            strides,
            weights_regularizer=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001,
            projection_shortcut=None
    ):
        super().__init__()

        self.projection_shortcut = projection_shortcut

        self.norm1 = BatchNormalization(
            axis=3,
            center=True,
            scale=True,
            fused=True,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )

        self.activation1 = Activation('relu')

        self.conv1 = Conv2DLayerFixedPadding(
            num_filters=num_filters,
            filter_size=3,
            strides=strides,
            weights_regularizer=weights_regularizer
        )

        self.norm2 = BatchNormalization(
            axis=3,
            center=True,
            scale=True,
            fused=True,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )

        self.activation2 = Activation('relu')

        self.conv2 = Conv2DLayerFixedPadding(
            num_filters=num_filters,
            filter_size=3,
            strides=1,
            weights_regularizer=weights_regularizer
        )

        for layer in [self.norm1, self.activation1, self.conv1,
                      self.norm2, self.activation2, self.conv2]:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        shortcut = inputs

        hidden = self.norm1(inputs, training=training)
        hidden = self.activation1(hidden, training=training)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(hidden, training=training)

        hidden = self.conv1(hidden, training=training)
        hidden = self.norm2(hidden, training=training)
        hidden = self.activation2(hidden, training=training)
        hidden = self.conv2(hidden, training=training)

        return hidden + shortcut


class ResNetBottleneckBlock(LudwigModule):

    def __init__(
            self,
            num_filters,
            strides,
            weights_regularizer=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001,
            projection_shortcut=None
    ):
        super().__init__()

        self.projection_shortcut = projection_shortcut

        self.norm1 = BatchNormalization(
            axis=3,
            center=True,
            scale=True,
            fused=True,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )

        self.activation1 = Activation('relu')

        self.conv1 = Conv2DLayerFixedPadding(
            num_filters=num_filters,
            filter_size=1,
            strides=1,
            weights_regularizer=weights_regularizer
        )

        self.norm2 = BatchNormalization(
            axis=3,
            center=True,
            scale=True,
            fused=True,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )

        self.activation2 = Activation('relu')

        self.conv2 = Conv2DLayerFixedPadding(
            num_filters=num_filters,
            filter_size=3,
            strides=strides,
            weights_regularizer=weights_regularizer
        )

        self.norm3 = BatchNormalization(
            axis=3,
            center=True,
            scale=True,
            fused=True,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )

        self.activation3 = Activation('relu')

        self.conv3 = Conv2DLayerFixedPadding(
            num_filters=4 * num_filters,
            filter_size=1,
            strides=1,
            weights_regularizer=weights_regularizer
        )

        for layer in [self.norm1, self.activation1, self.conv1,
                      self.norm2, self.activation2, self.conv2,
                      self.norm3, self.activation3, self.conv3]:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        shortcut = inputs

        hidden = self.norm1(inputs, training=training)
        hidden = self.activation1(hidden, training=training)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if self.projection_shortcut is not None:
            shortcut = self.projection_shortcut(hidden, training=training)

        hidden = self.conv1(hidden, training=training)
        hidden = self.norm2(hidden, training=training)
        hidden = self.activation2(hidden, training=training)
        hidden = self.conv2(hidden, training=training)
        hidden = self.norm3(hidden, training=training)
        hidden = self.activation3(hidden, training=training)
        hidden = self.conv3(hidden, training=training)

        return hidden + shortcut


class ResNetBlockLayer(LudwigModule):
    def __init__(
            self,
            num_filters,
            is_bottleneck,
            block_fn,
            num_blocks,
            strides,
            weights_regularizer=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001
    ):
        super().__init__()
        # Bottleneck blocks end with 4x the number of filters as they start with
        num_filters_out = num_filters * 4 if is_bottleneck else num_filters

        projection_shortcut = Conv2DLayerFixedPadding(
            num_filters=num_filters_out,
            filter_size=1,
            strides=strides,
            weights_regularizer=weights_regularizer
        )

        self.layers = [
            block_fn(
                num_filters,
                strides,
                weights_regularizer=weights_regularizer,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_epsilon=batch_norm_epsilon,
                projection_shortcut=projection_shortcut
            )
        ]

        for _ in range(1, num_blocks):
            self.layers.append(
                block_fn(
                    num_filters,
                    1,
                    weights_regularizer=weights_regularizer,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_epsilon=batch_norm_epsilon
                )
            )

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden, training=training)

        return hidden


class ResNet(LudwigModule):
    def __init__(
            self,
            resnet_size,
            is_bottleneck,
            num_filters,
            filter_size,
            conv_stride,
            first_pool_size,
            first_pool_stride,
            block_sizes,
            block_strides,
            weights_regularizer=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001
    ):
        """Creates a model obtaining an image representation.

         Implements ResNet v2:
         Identity Mappings in Deep Residual Networks
         https://arxiv.org/pdf/1603.05027.pdf
         by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

         Args:
           resnet_size: A single integer for the size of the ResNet model.
           is_bottleneck: Use regular blocks or bottleneck blocks.
           num_filters: The number of filters to use for the first block layer
             of the model. This number is then doubled for each subsequent block
             layer.
           filter_size: The kernel size to use for convolution.
           conv_stride: stride size for the initial convolutional layer
           first_pool_size: Pool size to be used for the first pooling layer.
             If none, the first pooling layer is skipped.
           first_pool_stride: stride size for the first pooling layer. Not used
             if first_pool_size is None.
           block_sizes: A list containing n values, where n is the number of sets of
             block layers desired. Each value should be the number of blocks in the
             i-th set.
           block_strides: List of integers representing the desired stride size for
             each of the sets of block layers. Should be same length as block_sizes.
         Raises:
           ValueError: if invalid version is selected.
        """
        super().__init__()
        self.resnet_size = resnet_size

        block_class = ResNetBlock
        if is_bottleneck:
            block_class = ResNetBottleneckBlock

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.pre_activation = True
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon

        self.layers = [
            Conv2DLayerFixedPadding(
                num_filters=num_filters,
                filter_size=filter_size,
                strides=conv_stride,
                weights_regularizer=weights_regularizer
            )
        ]
        if first_pool_size:
            self.layers.append(
                MaxPool2D(
                    pool_size=first_pool_size,
                    strides=first_pool_stride,
                    padding='same'
                )
            )
        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2 ** i)
            self.layers.append(
                ResNetBlockLayer(
                    num_filters,
                    is_bottleneck,
                    block_class,
                    num_blocks,
                    block_strides[i],
                    weights_regularizer=weights_regularizer,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_epsilon=batch_norm_epsilon
                )
            )
        if self.pre_activation:
            self.layers.append(
                BatchNormalization(
                    axis=3,
                    center=True,
                    scale=True,
                    fused=True,
                    momentum=batch_norm_momentum,
                    epsilon=batch_norm_epsilon
                )
            )
            self.layers.append(Activation('relu'))

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.layers:
            hidden = layer(hidden, training=training)

        axes = [1, 2]
        hidden = math.reduce_mean(hidden, axes, keepdims=True)
        hidden = squeeze(hidden, axes)

        return hidden


################################################################################
# The following code for ResNet is adapted from the TensorFlow implementation
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
################################################################################

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
resnet_choices = {
    8: [1, 2, 2],
    14: [1, 2, 2],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


def get_resnet_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
    Returns:
      A list of block sizes to use in building the model.
    Raises:
      KeyError: if invalid resnet_size is received.
    """
    try:
        return resnet_choices[resnet_size]
    except KeyError:
        err = (
            'Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, resnet_choices.keys()
            )
        )
        raise ValueError(err)
