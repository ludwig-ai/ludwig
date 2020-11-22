#! /usr/bin/env python
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
from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Flatten

from ludwig.encoders.base import Encoder
from ludwig.utils.registry import Registry, register, register_default
from ludwig.modules.convolutional_modules import Conv2DStack, \
    get_resnet_block_sizes
from ludwig.modules.convolutional_modules import ResNet2
from ludwig.modules.fully_connected_modules import FCStack

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry()


class ImageEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register_default(name='stacked_cnn')
class Stacked2DCNN(ImageEncoder):

    def __init__(
            self,
            conv_layers=None,
            num_conv_layers=None,
            filter_size=3,
            num_filters=32,
            strides=(1, 1),
            padding='valid',
            dilation_rate=(1, 1),
            conv_use_bias=True,
            conv_weights_initializer='glorot_uniform',
            conv_bias_initializer='zeros',
            conv_weights_regularizer=None,
            conv_bias_regularizer=None,
            conv_activity_regularizer=None,
            # conv_weights_constraint=None,
            # conv_bias_constraint=None,
            conv_norm=None,
            conv_norm_params=None,
            conv_activation='relu',
            conv_dropout=0,
            pool_function='max',
            pool_size=(2, 2),
            pool_strides=None,
            fc_layers=None,
            num_fc_layers=1,
            fc_size=128,
            fc_use_bias=True,
            fc_weights_initializer='glorot_uniform',
            fc_bias_initializer='zeros',
            fc_weights_regularizer=None,
            fc_bias_regularizer=None,
            fc_activity_regularizer=None,
            # fc_weights_constraint=None,
            # fc_bias_constraint=None,
            fc_norm=None,
            fc_norm_params=None,
            fc_activation='relu',
            fc_dropout=0,
            **kwargs
    ):
        super(Stacked2DCNN, self).__init__()

        logger.debug(' {}'.format(self.name))

        logger.debug('  Conv2DStack')
        self.conv_stack_2d = Conv2DStack(
            layers=conv_layers,
            num_layers=num_conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_strides=strides,
            default_padding=padding,
            default_dilation_rate=dilation_rate,
            default_use_bias=conv_use_bias,
            default_weights_initializer=conv_weights_initializer,
            default_bias_initializer=conv_bias_initializer,
            default_weights_regularizer=conv_weights_regularizer,
            default_bias_regularizer=conv_bias_regularizer,
            default_activity_regularizer=conv_activity_regularizer,
            # default_weights_constraint=conv_weights_constraint,
            # default_bias_constraint=conv_bias_constraint,
            default_norm=conv_norm,
            default_norm_params=conv_norm_params,
            default_activation=conv_activation,
            default_dropout=conv_dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_strides=pool_strides,
        )

        logger.debug('  FCStacl')
        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=fc_use_bias,
            default_weights_initializer=fc_weights_initializer,
            default_bias_initializer=fc_bias_initializer,
            default_weights_regularizer=fc_weights_regularizer,
            default_bias_regularizer=fc_bias_regularizer,
            default_activity_regularizer=fc_activity_regularizer,
            # default_weights_constraint=fc_weights_constraint,
            # default_bias_constraint=fc_bias_constraint,
            default_norm=fc_norm,
            default_norm_params=fc_norm_params,
            default_activation=fc_activation,
            default_dropout=fc_dropout,
        )

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                    Shape: [batch x height x width x channels], type tf.uint8
        """

        # ================ Conv Layers ================
        hidden = self.conv_stack_2d(
            inputs,
            training,
        )
        hidden = tf.reshape(hidden, [hidden.shape[0], -1])

        # ================ Fully Connected ================
        outputs = self.fc_stack(hidden)

        return {'encoder_output': outputs}


@register(name='resnet')
class ResNetEncoder(ImageEncoder):

    def __init__(
            self,
            resnet_size=50,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            batch_norm_momentum=0.9,
            batch_norm_epsilon=0.001,
            fc_layers=None,
            num_fc_layers=1,
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
            **kwargs
    ):
        super(ResNetEncoder, self).__init__()
        logger.debug(' {}'.format(self.name))

        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        block_sizes = get_resnet_block_sizes(resnet_size)
        block_strides = [1, 2, 2, 2][:len(block_sizes)]

        logger.debug('  ResNet2')
        self.resnet = ResNet2(
            resnet_size,
            bottleneck,
            num_filters,
            kernel_size,
            conv_stride,
            first_pool_size,
            first_pool_stride,
            block_sizes,
            block_strides,
            batch_norm_momentum,
            batch_norm_epsilon
        )

        self.flatten = Flatten()

        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=fc_weights_constraint,
            # default_bias_constraint=fc_bias_constraint,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
        )

    def call(self, inputs, training=None, mask=None):

        hidden = self.resnet(inputs, training=training)
        hidden = self.flatten(hidden, training=training)
        hidden = self.fc_stack(hidden, training=training)

        return {'encoder_output': hidden}
