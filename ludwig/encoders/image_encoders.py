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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ludwig.encoders.base import Encoder
from ludwig.modules.mlp_mixer_modules import MLPMixer
from ludwig.utils.registry import Registry, register, register_default
from ludwig.modules.convolutional_modules import Conv2DStack, ResNet,\
    get_resnet_block_sizes
from ludwig.modules.fully_connected_modules import FCStack

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry()


class ImageEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


# TODO(shreya): Add type hints for missing args
@register_default(name='stacked_cnn')
class Stacked2DCNN(ImageEncoder):

    def __init__(
            self,
            img_height: int,
            img_width: int,
            conv_layers: Optional[List[Dict]] = None,
            num_conv_layers: Optional[int] = None,
            first_in_channels: int = None,
            out_channels: int = 32,
            kernel_size: Union[int, Tuple[int]] = 3,
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int], str] = 'valid',
            dilation: Union[int, Tuple[int]] = 1,
            conv_bias: bool = True,
            padding_mode: str ='zeros',
            conv_norm: Optional[str] = None,
            conv_norm_params: Optional[Dict[str, Any]] = None,
            conv_activation: str = 'relu',
            conv_dropout: int = 0,
            pool_function: str = 'max',
            pool_kernel_size: Union[int, Tuple[int]] = 2,
            pool_stride: Union[int, Tuple[int]] = None,
            pool_padding: Union[int, Tuple[int], str]  ='valid',
            pool_dilation: Union[int, Tuple[int]] = 1,
            groups: int = 1,
            fc_layers: Optional[List[Dict]] = None,
            num_fc_layers: Optional[int] = 1,
            fc_size: int = 128,
            fc_use_bias: bool = True,
            fc_weights_initializer: str = 'glorot_uniform',
            fc_bias_initializer: str = 'zeros',
            fc_weights_regularizer=None,
            fc_bias_regularizer=None,
            fc_activity_regularizer=None,
            fc_norm: Optional[str] = None,
            fc_norm_params: Optional[Dict[str, Any]] = None,
            fc_activation: str = 'relu',
            fc_dropout: float = 0,
            **kwargs
    ):
        super().__init__()

        logger.debug(' {}'.format(self.name))

        if first_in_channels is None:
            raise ValueError('first_in_channels must not be None.')

        logger.debug('  Conv2DStack')
        self.conv_stack_2d = Conv2DStack(
            img_height=img_height,
            img_width=img_width,
            layers=conv_layers,
            num_layers=num_conv_layers,
            first_in_channels=first_in_channels,
            default_out_channels=out_channels,
            default_kernel_size=kernel_size,
            default_stride=stride,
            default_padding=padding,
            default_dilation=dilation,
            default_groups=groups,
            default_bias=conv_bias,
            default_padding_mode=padding_mode,
            default_norm=conv_norm,
            default_norm_params=conv_norm_params,
            default_activation=conv_activation,
            default_dropout=conv_dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_kernel_size,
            default_pool_stride=pool_stride,
            default_pool_padding=pool_padding,
            default_pool_dilation=pool_dilation,
        )
        out_channels, img_height, img_width = self.conv_stack_2d.output_shape
        first_fc_layer_input_size = out_channels * img_height * img_width

        self.flatten = torch.nn.Flatten()

        logger.debug('  FCStack')
        # TODO(shreya): Confirm that FCStack params are OK.
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=fc_use_bias,
            default_weights_initializer=fc_weights_initializer,
            default_bias_initializer=fc_bias_initializer,
            default_weights_regularizer=fc_weights_regularizer,
            default_bias_regularizer=fc_bias_regularizer,
            default_activity_regularizer=fc_activity_regularizer,
            default_norm=fc_norm,
            default_norm_params=fc_norm_params,
            default_activation=fc_activation,
            default_dropout=fc_dropout,
        )

    def forward(self, inputs):
        """
            :param inputs: The inputs fed into the encoder.
                    Shape: [batch x channels x height x width], type torch.uint8
        """

        # ================ Conv Layers ================
        hidden = self.conv_stack_2d(
            inputs,
        )
        hidden = self.flatten(hidden)

        # ================ Fully Connected ================
        outputs = self.fc_stack(hidden)

        return {'encoder_output': outputs}

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


# TODO(shreya): Add type hints for missing args.
@register(name='resnet')
class ResNetEncoder(ImageEncoder):

    def __init__(
            self,
            img_height: int,
            img_width: int,
            resnet_size: int = 50,
            first_in_channels: int = 3,
            out_channels: int = 16,
            kernel_size: Union[int, Tuple[int]] = 3,
            conv_stride: Union[int, Tuple[int]] = 1,
            first_pool_kernel_size: Union[int, Tuple[int]] = None,
            first_pool_stride: Union[int, Tuple[int]] = None,
            batch_norm_momentum: float = 0.9,
            batch_norm_epsilon: float = 0.001,
            fc_layers: Optional[List[Dict]] = None,
            num_fc_layers: Optional[int] = 1,
            fc_size: int = 256,
            use_bias: bool = True,
            weights_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            norm: Optional[str] = None,
            norm_params: Optional[Dict[str, Any]] = None,
            activation: str = 'relu',
            dropout: float = 0,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        if resnet_size < 50:
            bottleneck = False
        else:
            bottleneck = True

        block_sizes = get_resnet_block_sizes(resnet_size)
        block_strides = [1, 2, 2, 2][:len(block_sizes)]

        logger.debug('  ResNet')
        self.resnet = ResNet(
            img_height=img_height,
            img_width=img_width,
            resnet_size=resnet_size,
            is_bottleneck=bottleneck,
            first_in_channels=first_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            first_pool_kernel_size=first_pool_kernel_size,
            first_pool_stride=first_pool_stride,
            block_sizes=block_sizes,
            block_strides=block_strides,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon
        )
        out_channels, img_height, img_width = self.resnet.output_shape
        first_fc_layer_input_size = out_channels * img_height * img_width

        self.flatten = torch.nn.Flatten()

        # TODO(shreya): Confirm that arguments of FCStack are OK.
        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
        )

    def forward(self, inputs):

        hidden = self.resnet(inputs)
        hidden = self.flatten(hidden)
        hidden = self.fc_stack(hidden)

        return {'encoder_output': hidden}

    def get_output_shape(self, input_shape):
        # TODO(shreya): Confirm that this is implemented
        return self.fc_stack.output_shape


@register(name='mlp_mixer')
class MLPMixerEncoder(ImageEncoder):

    def __init__(
            self,
            img_height: int,
            img_width: int,
            in_channels: int,
            patch_size: int = 16,
            embed_size: int = 512,
            token_size: int = 2048,
            channel_dim: int = 256,
            num_layers: int = 8,
            dropout: float = 0.0,
            avg_pool: bool = True,
            **kwargs
    ):
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self._input_shape = (in_channels, img_height, img_width)

        logger.debug('  MLPMixer')
        self.mlp_mixer = MLPMixer(
            img_height=img_height,
            img_width=img_width,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_size=embed_size,
            token_size=token_size,
            channel_dim=channel_dim,
            num_layers=num_layers,
            dropout=dropout,
            avg_pool=avg_pool,
        )

        self._output_shape = self.mlp_mixer.output_shape

    def forward(self, inputs):
        hidden = self.mlp_mixer(inputs)
        return {'encoder_output': hidden}

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape
