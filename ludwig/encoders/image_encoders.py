#! /usr/bin/env python
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
from typing import Dict

import torch

from ludwig.constants import IMAGE
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.convolutional_modules import Conv2DStack, ResNet
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.mlp_mixer_modules import MLPMixer
from ludwig.schema.encoders.image_encoders import (
    MLPMixerEncoderConfig,
    ResNetEncoderConfig,
    Stacked2DCNNEncoderConfig,
    ViTEncoderConfig,
)
from ludwig.utils.pytorch_utils import freeze_parameters

logger = logging.getLogger(__name__)


# TODO(shreya): Add type hints for missing args
@register_encoder("stacked_cnn", IMAGE)
class Stacked2DCNN(Encoder):
    def __init__(self, encoder_config: Stacked2DCNNEncoderConfig = Stacked2DCNNEncoderConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")

        # map parameter input feature config names to internal names
        img_height = encoder_config.height
        img_width = encoder_config.width
        first_in_channels = encoder_config.num_channels

        self._input_shape = (first_in_channels, img_height, img_width)

        if first_in_channels is None:
            raise ValueError("first_in_channels must not be None.")

        logger.debug("  Conv2DStack")
        self.conv_stack_2d = Conv2DStack(
            img_height=img_height,
            img_width=img_width,
            layers=encoder_config.conv_layers,
            num_layers=encoder_config.num_conv_layers,
            first_in_channels=first_in_channels,
            default_out_channels=encoder_config.out_channels,
            default_kernel_size=encoder_config.kernel_size,
            default_stride=encoder_config.stride,
            default_padding=encoder_config.padding,
            default_dilation=encoder_config.dilation,
            default_groups=encoder_config.groups,
            default_use_bias=encoder_config.conv_use_bias,
            default_padding_mode=encoder_config.padding_mode,
            default_norm=encoder_config.conv_norm,
            default_norm_params=encoder_config.conv_norm_params,
            default_activation=encoder_config.conv_activation,
            default_dropout=encoder_config.conv_dropout,
            default_pool_function=encoder_config.pool_function,
            default_pool_kernel_size=encoder_config.pool_kernel_size,
            default_pool_stride=encoder_config.pool_stride,
            default_pool_padding=encoder_config.pool_padding,
            default_pool_dilation=encoder_config.pool_dilation,
        )
        out_channels, img_height, img_width = self.conv_stack_2d.output_shape
        first_fc_layer_input_size = out_channels * img_height * img_width

        self.flatten = torch.nn.Flatten()

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=encoder_config.fc_layers,
            num_layers=encoder_config.num_fc_layers,
            default_output_size=encoder_config.output_size,
            default_use_bias=encoder_config.fc_use_bias,
            default_weights_initializer=encoder_config.fc_weights_initializer,
            default_bias_initializer=encoder_config.fc_bias_initializer,
            default_norm=encoder_config.fc_norm,
            default_norm_params=encoder_config.fc_norm_params,
            default_activation=encoder_config.fc_activation,
            default_dropout=encoder_config.fc_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The inputs fed into the encoder.
                Shape: [batch x channels x height x width], type torch.uint8
        """

        hidden = self.conv_stack_2d(inputs)
        hidden = self.flatten(hidden)
        outputs = self.fc_stack(hidden)

        return {"encoder_output": outputs}

    @staticmethod
    def get_schema_cls():
        return Stacked2DCNNEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


@register_encoder("resnet", IMAGE)
class ResNetEncoder(Encoder):
    def __init__(self, encoder_config: ResNetEncoderConfig = ResNetEncoderConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")
        # map parameter input feature config names to internal names
        img_height = encoder_config.height
        img_width = encoder_config.width
        first_in_channels = encoder_config.num_channels

        self._input_shape = (first_in_channels, img_height, img_width)

        logger.debug("  ResNet")
        self.resnet = ResNet(
            img_height=img_height,
            img_width=img_width,
            first_in_channels=first_in_channels,
            out_channels=encoder_config.out_channels,
            resnet_size=encoder_config.resnet_size,
            kernel_size=encoder_config.kernel_size,
            conv_stride=encoder_config.conv_stride,
            first_pool_kernel_size=encoder_config.first_pool_kernel_size,
            first_pool_stride=encoder_config.first_pool_stride,
            batch_norm_momentum=encoder_config.batch_norm_momentum,
            batch_norm_epsilon=encoder_config.batch_norm_epsilon,
        )
        first_fc_layer_input_size = self.resnet.output_shape[0]

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=encoder_config.fc_layers,
            num_layers=encoder_config.num_fc_layers,
            default_output_size=encoder_config.output_size,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=encoder_config.norm,
            default_norm_params=encoder_config.norm_params,
            default_activation=encoder_config.activation,
            default_dropout=encoder_config.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:

        hidden = self.resnet(inputs)
        axes = [2, 3]
        hidden = torch.mean(hidden, axes)
        hidden = self.fc_stack(hidden)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return ResNetEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


@register_encoder("mlp_mixer", IMAGE)
class MLPMixerEncoder(Encoder):
    def __init__(self, encoder_config: MLPMixerEncoderConfig = MLPMixerEncoderConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")
        # map parameter input feature config names to internal names
        img_height = encoder_config.height
        img_width = encoder_config.width
        in_channels = encoder_config.num_channels

        if in_channels is None:
            raise RuntimeError("num_channels must not be None")

        self._input_shape = (in_channels, img_height, img_width)

        logger.debug("  MLPMixer")
        self.mlp_mixer = MLPMixer(
            img_height=img_height,
            img_width=img_width,
            in_channels=in_channels,
            patch_size=encoder_config.patch_size,
            embed_size=encoder_config.embed_size,
            token_size=encoder_config.token_size,
            channel_dim=encoder_config.channel_dim,
            num_layers=encoder_config.num_layers,
            dropout=encoder_config.dropout,
            avg_pool=encoder_config.avg_pool,
        )

        self._output_shape = self.mlp_mixer.output_shape

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.mlp_mixer(inputs)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return MLPMixerEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape


@register_encoder("vit", IMAGE)
class ViTEncoder(Encoder):
    def __init__(self, encoder_config: ViTEncoderConfig):
        """Creates a ViT encoder using transformers.ViTModel."""
        super().__init__(encoder_config)
        try:
            from transformers import ViTConfig, ViTModel
        except ModuleNotFoundError:
            raise RuntimeError(
                " transformers is not installed. "
                "In order to install all image feature dependencies run "
                "pip install ludwig[image]"
            )

        # map parameter input feature config names to internal names
        img_height = encoder_config.height
        img_width = encoder_config.width
        in_channels = encoder_config.num_channels

        img_width = img_width or img_height
        if img_width != img_height:
            raise ValueError("img_height and img_width should be identical.")
        self._input_shape = (in_channels, img_height, img_width)

        if encoder_config.use_pretrained and not encoder_config.saved_weights_in_checkpoint:
            self.transformer = ViTModel.from_pretrained(encoder_config.pretrained_model)
        else:
            config = ViTConfig(
                image_size=img_height,
                num_channels=in_channels,
                patch_size=encoder_config.patch_size,
                hidden_size=encoder_config.hidden_size,
                num_hidden_layers=encoder_config.num_hidden_layers,
                num_attention_heads=encoder_config.num_attention_heads,
                intermediate_size=encoder_config.intermediate_size,
                hidden_act=encoder_config.hidden_act,
                hidden_dropout_prob=encoder_config.hidden_dropout_prob,
                attention_probs_dropout_prob=encoder_config.attention_probs_dropout_prob,
                initializer_range=encoder_config.initializer_range,
                layer_norm_eps=encoder_config.layer_norm_eps,
                gradient_checkpointing=encoder_config.gradient_checkpointing,
            )
            self.transformer = ViTModel(config)

        if encoder_config.trainable:
            self.transformer.train()
        else:
            freeze_parameters(self.transformer)

        self._output_shape = (self.transformer.config.hidden_size,)
        self.output_attentions = encoder_config.output_attentions

    def forward(self, inputs: torch.Tensor, head_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        output = self.transformer(inputs, head_mask=head_mask, output_attentions=self.output_attentions)
        return_dict = {"encoder_output": output.pooler_output}
        if self.output_attentions:
            return_dict["attentions"] = output.attentions
        return return_dict

    @staticmethod
    def get_schema_cls():
        return ViTEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)
