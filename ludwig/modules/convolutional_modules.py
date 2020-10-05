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
from tensorflow import math, squeeze
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Activation, AveragePooling1D,
                                     AveragePooling2D, BatchNormalization,
                                     Conv1D, Conv2D, Dropout, Layer,
                                     LayerNormalization, MaxPool1D, MaxPool2D,
                                     ZeroPadding2D)

logger = logging.getLogger(__name__)


class Conv1DLayer(Layer):

    def __init__(
            self,
            num_filters=256,
            filter_size=3,
            strides=1,
            padding='same',
            dilation_rate=1,
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
            pool_size=2,
            pool_strides=None,
            pool_padding='valid',
    ):
        super(Conv1DLayer, self).__init__()

        self.layers = []

        self.layers.append(Conv1D(
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
            # kernel_constraint=None,
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
            pool = MaxPool1D
            if pool_function in {'average', 'avg', 'mean'}:
                pool = AveragePooling1D
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


class Conv1DStack(Layer):

    def __init__(
            self,
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
            default_pool_padding='same',
            **kwargs
    ):
        super(Conv1DStack, self).__init__()

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

        for i, layer in enumerate(self.layers):
            logger.debug('   stack layer {}'.format(i))
            self.stack.append(
                Conv1DLayer(
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

        if hidden.shape[1] == 0:
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


class ParallelConv1D(Layer):

    def __init__(
            self,
            layers=None,
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
            default_pool_size=None,
            default_pool_strides=None,
            default_pool_padding='valid',
            **kwargs
    ):
        super(ParallelConv1D, self).__init__()

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
        hiddens = []

        for layer in self.parallel_layers:
            hiddens.append(layer(hidden, training=training))
        hidden = tf.concat(hiddens, 2)

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

        return hidden


class ParallelConv1DStack(Layer):

    def __init__(
            self,
            stacked_layers=None,
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
        super(ParallelConv1DStack, self).__init__()

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

        for i, parallel_layers in enumerate(self.stacked_parallel_layers):
            logger.debug('   stack layer {}'.format(i))
            self.stack.append(ParallelConv1D(layers=parallel_layers))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs

        for layer in self.stack:
            hidden = layer(hidden, training=training)

        if hidden.shape[1] == 0:
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


class Conv2DLayer(Layer):

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
        super(Conv2DLayer, self).__init__()

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


class Conv2DStack(Layer):

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
        super(Conv2DStack, self).__init__()

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


class Conv2DLayerFixedPadding(Layer):

    def __init__(
            self,
            num_filters=256,
            filter_size=3,
            strides=1,
            weights_regularizer=None
    ):
        super(Conv2DLayerFixedPadding, self).__init__()

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


class ResNetBlock(Layer):

    def __init__(
            self,
            num_filters,
            strides,
            weights_regularizer=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001,
            projection_shortcut=None
    ):
        super(ResNetBlock, self).__init__()

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


class ResNetBottleneckBlock(Layer):

    def __init__(
            self,
            num_filters,
            strides,
            weights_regularizer=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001,
            projection_shortcut=None
    ):
        super(ResNetBottleneckBlock, self).__init__()

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


class ResNetBlockLayer(Layer):
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
        super(ResNetBlockLayer, self).__init__()
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


class ResNet2(Layer):
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
        super(ResNet2, self).__init__()
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
# def resnet_batch_norm(inputs, is_training,
#                       batch_norm_momentum=0.9, batch_norm_epsilon=0.001):
#     """Performs a batch normalization using a standard set of parameters."""
#     # We set fused=True for a significant performance boost. See
#     # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
#     # Original implementation default values:
#     # _BATCH_NORM_DECAY = 0.997
#     # _BATCH_NORM_EPSILON = 1e-5
#     # they lead to a big difference between the loss
#     # at train and prediction time
#     return tf.layers.batch_normalization(
#         inputs=inputs, axis=3,
#         momentum=batch_norm_momentum, epsilon=batch_norm_epsilon, center=True,
#         scale=True, training=is_training, fused=True)
#
#
# def fixed_padding(inputs, kernel_size):
#     """Pads the input along the spatial dimensions independently of input size.
#     Args:
#       inputs: A tensor of size [batch, channels, height_in, width_in] or
#         [batch, height_in, width_in, channels] depending on data_format.
#       kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
#                    Should be a positive integer.
#     Returns:
#       A tensor with the same format as the input with the data either intact
#       (if kernel_size == 1) or padded (if kernel_size > 1).
#     """
#     pad_total = kernel_size - 1
#     pad_beg = pad_total // 2
#     pad_end = pad_total - pad_beg
#     padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
#                                     [pad_beg, pad_end], [0, 0]])
#     return padded_inputs
#
#
# def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
#                          regularizer=None):
#     """Strided 2-D convolution with explicit padding."""
#     # The padding is consistent and is based only on `kernel_size`, not on the
#     # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
#     if strides > 1:
#         inputs = fixed_padding(inputs, kernel_size)
#
#     return tf.layers.conv2d(
#         inputs=inputs, filters=filters, kernel_size=kernel_size,
#         strides=strides,
#         padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
#         kernel_initializer=tf.variance_scaling_initializer(),
#         kernel_regularizer=regularizer)


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

# ################################################################################
# # ResNet block configs.
# ################################################################################
# def resnet_block(inputs, filters, is_training, projection_shortcut, strides,
#                  regularizer=None, batch_norm_momentum=0.9,
#                  batch_norm_epsilon=0.001):
#     """A single block for ResNet v2, without a bottleneck.
#     Batch normalization then ReLu then convolution as described by:
#       Identity Mappings in Deep Residual Networks
#       https://arxiv.org/pdf/1603.05027.pdf
#       by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
#     Args:
#       inputs: A tensor of size [batch, channels, height_in, width_in] or
#         [batch, height_in, width_in, channels] depending on data_format.
#       filters: The number of filters for the convolutions.
#       is_training: A Boolean for whether the model is in training or inference
#         mode. Needed for batch normalization.
#       projection_shortcut: The function to use for projection shortcuts
#         (typically a 1x1 convolution when downsampling the input).
#       strides: The block's stride. If greater than 1, this block will ultimately
#         downsample the input.
#     Returns:
#       The output tensor of the block; shape should match inputs.
#     """
#     shortcut = inputs
#     inputs = resnet_batch_norm(inputs, is_training,
#                                batch_norm_momentum=batch_norm_momentum,
#                                batch_norm_epsilon=batch_norm_epsilon)
#     inputs = tf.nn.relu(inputs)
#
#     # The projection shortcut should come after the first batch norm and ReLU
#     # since it performs a 1x1 convolution.
#     if projection_shortcut is not None:
#         shortcut = projection_shortcut(inputs)
#
#     inputs = conv2d_fixed_padding(
#         inputs=inputs, filters=filters, kernel_size=3, strides=strides,
#         regularizer=regularizer)
#
#     inputs = resnet_batch_norm(inputs, is_training,
#                                batch_norm_momentum=batch_norm_momentum,
#                                batch_norm_epsilon=batch_norm_epsilon)
#     inputs = tf.nn.relu(inputs)
#     inputs = conv2d_fixed_padding(
#         inputs=inputs, filters=filters, kernel_size=3, strides=1,
#         regularizer=regularizer)
#
#     return inputs + shortcut
#
#
# def resnet_bottleneck_block(inputs, filters, is_training, projection_shortcut,
#                             strides, regularizer=None, batch_norm_momentum=0.9,
#                             batch_norm_epsilon=0.001):
#     """A single block for ResNet v2, with a bottleneck.
#     Similar to _building_block_v2(), except using the "bottleneck" blocks
#     described in:
#       Convolution then batch normalization then ReLU as described by:
#         Deep Residual Learning for Image Recognition
#         https://arxiv.org/pdf/1512.03385.pdf
#         by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
#     Adapted to the ordering conventions of:
#       Batch normalization then ReLu then convolution as described by:
#         Identity Mappings in Deep Residual Networks
#         https://arxiv.org/pdf/1603.05027.pdf
#         by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
#     Args:
#       inputs: A tensor of size [batch, channels, height_in, width_in] or
#         [batch, height_in, width_in, channels] depending on data_format.
#       filters: The number of filters for the convolutions.
#       is_training: A Boolean for whether the model is in training or inference
#         mode. Needed for batch normalization.
#       projection_shortcut: The function to use for projection shortcuts
#         (typically a 1x1 convolution when downsampling the input).
#       strides: The block's stride. If greater than 1, this block will ultimately
#         downsample the input.
#     Returns:
#       The output tensor of the block; shape should match inputs.
#     """
#     shortcut = inputs
#     inputs = resnet_batch_norm(inputs, is_training,
#                                batch_norm_momentum=batch_norm_momentum,
#                                batch_norm_epsilon=batch_norm_epsilon)
#     inputs = tf.nn.relu(inputs)
#
#     # The projection shortcut should come after the first batch norm and ReLU
#     # since it performs a 1x1 convolution.
#     if projection_shortcut is not None:
#         shortcut = projection_shortcut(inputs)
#
#     inputs = conv2d_fixed_padding(
#         inputs=inputs, filters=filters, kernel_size=1, strides=1,
#         regularizer=regularizer)
#
#     inputs = resnet_batch_norm(inputs, is_training,
#                                batch_norm_momentum=batch_norm_momentum,
#                                batch_norm_epsilon=batch_norm_epsilon)
#     inputs = tf.nn.relu(inputs)
#     inputs = conv2d_fixed_padding(
#         inputs=inputs, filters=filters, kernel_size=3, strides=strides,
#         regularizer=regularizer)
#
#     inputs = resnet_batch_norm(inputs, is_training,
#                                batch_norm_momentum=batch_norm_momentum,
#                                batch_norm_epsilon=batch_norm_epsilon)
#     inputs = tf.nn.relu(inputs)
#     inputs = conv2d_fixed_padding(
#         inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
#         regularizer=regularizer)
#
#     return inputs + shortcut
#
#
# def resnet_block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
#                        is_training, name, regularizer=None,
#                        batch_norm_momentum=0.9, batch_norm_epsilon=0.001):
#     """Creates one layer of blocks for the ResNet model.
#     Args:
#       inputs: A tensor of size [batch, channels, height_in, width_in] or
#         [batch, height_in, width_in, channels] depending on data_format.
#       filters: The number of filters for the first convolution of the layer.
#       bottleneck: Is the block created a bottleneck block.
#       block_fn: The block to use within the model, either `building_block` or
#         `bottleneck_block`.
#       blocks: The number of blocks contained in the layer.
#       strides: The stride to use for the first convolution of the layer. If
#         greater than 1, this layer will ultimately downsample the input.
#       is_training: Either True or False, whether we are currently training the
#         model. Needed for batch norm.
#       name: A string name for the tensor output of the block layer.
#     Returns:
#       The output tensor of the block layer.
#     """
#
#     # Bottleneck blocks end with 4x the number of filters as they start with
#     filters_out = filters * 4 if bottleneck else filters
#
#     def projection_shortcut(inputs):
#         return conv2d_fixed_padding(
#             inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
#             regularizer=regularizer)
#
#     # Only the first block per block_layer uses projection_shortcut and strides
#     inputs = block_fn(inputs, filters, is_training, projection_shortcut,
#                       strides, regularizer=regularizer,
#                       batch_norm_momentum=batch_norm_momentum,
#                       batch_norm_epsilon=batch_norm_epsilon)
#
#     for _ in range(1, blocks):
#         inputs = block_fn(inputs, filters, is_training, None, 1,
#                           regularizer=regularizer,
#                           batch_norm_momentum=batch_norm_momentum,
#                           batch_norm_epsilon=batch_norm_epsilon)
#
#     return tf.identity(inputs, name)
#
#
# class ResNet(object):
#     """Base class for building the Resnet Model."""
#
#     def __init__(self, resnet_size, bottleneck, num_filters,
#                  kernel_size, conv_stride, first_pool_size, first_pool_stride,
#                  block_sizes, block_strides, batch_norm_momentum=0.9,
#                  batch_norm_epsilon=0.001):
#         """Creates a model obtaining an image representation.
#
#         Implements ResNet v2:
#         Identity Mappings in Deep Residual Networks
#         https://arxiv.org/pdf/1603.05027.pdf
#         by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
#
#         Args:
#           resnet_size: A single integer for the size of the ResNet model.
#           bottleneck: Use regular blocks or bottleneck blocks.
#           num_filters: The number of filters to use for the first block layer
#             of the model. This number is then doubled for each subsequent block
#             layer.
#           kernel_size: The kernel size to use for convolution.
#           conv_stride: stride size for the initial convolutional layer
#           first_pool_size: Pool size to be used for the first pooling layer.
#             If none, the first pooling layer is skipped.
#           first_pool_stride: stride size for the first pooling layer. Not used
#             if first_pool_size is None.
#           block_sizes: A list containing n values, where n is the number of sets of
#             block layers desired. Each value should be the number of blocks in the
#             i-th set.
#           block_strides: List of integers representing the desired stride size for
#             each of the sets of block layers. Should be same length as block_sizes.
#         Raises:
#           ValueError: if invalid version is selected.
#         """
#         self.resnet_size = resnet_size
#
#         self.bottleneck = bottleneck
#         if bottleneck:
#             self.block_fn = resnet_bottleneck_block
#         else:
#             self.block_fn = resnet_block
#
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#         self.conv_stride = conv_stride
#         self.first_pool_size = first_pool_size
#         self.first_pool_stride = first_pool_stride
#         self.block_sizes = block_sizes
#         self.block_strides = block_strides
#         self.pre_activation = True
#         self.batch_norm_momentum = batch_norm_momentum
#         self.batch_norm_epsilon = batch_norm_epsilon
#
#     def __call__(
#             self,
#             input_image,
#             regularizer,
#             dropout,
#             is_training=True
#     ):
#         """Add operations to classify a batch of input images.
#         Args:
#           input_image: A Tensor representing a batch of input images.
#           is_training: A boolean. Set to True to add operations required only when
#             training the classifier.
#         Returns:
#           A logits Tensor with shape [<batch_size>, <final_channels>].
#         """
#         inputs = input_image
#
#         with tf.variable_scope('resnet'):
#             inputs = conv2d_fixed_padding(
#                 inputs=inputs, filters=self.num_filters,
#                 kernel_size=self.kernel_size,
#                 strides=self.conv_stride, regularizer=regularizer)
#             inputs = tf.identity(inputs, 'initial_conv')
#
#             if self.first_pool_size:
#                 inputs = tf.layers.max_pooling2d(
#                     inputs=inputs, pool_size=self.first_pool_size,
#                     strides=self.first_pool_stride, padding='SAME')
#                 inputs = tf.identity(inputs, 'initial_max_pool')
#
#             for i, num_blocks in enumerate(self.block_sizes):
#                 num_filters = self.num_filters * (2 ** i)
#                 inputs = resnet_block_layer(
#                     inputs=inputs, filters=num_filters,
#                     bottleneck=self.bottleneck,
#                     block_fn=self.block_fn, blocks=num_blocks,
#                     strides=self.block_strides[i], is_training=is_training,
#                     name='block_layer{}'.format(i + 1),
#                     regularizer=regularizer,
#                     batch_norm_momentum=self.batch_norm_momentum,
#                     batch_norm_epsilon=self.batch_norm_epsilon
#                 )
#
#             # Only apply the BN and ReLU for model that does pre_activation in each
#             # building/bottleneck block, eg resnet V2.
#             if self.pre_activation:
#                 inputs = resnet_batch_norm(
#                     inputs, is_training,
#                     batch_norm_momentum=self.batch_norm_momentum,
#                     batch_norm_epsilon=self.batch_norm_epsilon)
#                 inputs = tf.nn.relu(inputs)
#
#             # The current top layer has shape
#             # `batch_size x pool_size x pool_size x final_size`.
#             # ResNet does an Average Pooling layer over pool_size,
#             # but that is the same as doing a reduce_mean. We do a reduce_mean
#             # here because it performs better than AveragePooling2D.
#             axes = [1, 2]
#             inputs = tf.reduce_mean(inputs, axes, keepdims=True)
#             inputs = tf.identity(inputs, 'final_reduce_mean')
#
#             inputs = tf.squeeze(inputs, axes)
#             return inputs
