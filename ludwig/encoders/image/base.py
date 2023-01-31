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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.convolutional_modules import Conv2DStack, ResNet
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.mlp_mixer_modules import MLPMixer
from ludwig.schema.encoders.image.base import MLPMixerConfig, ResNetConfig, Stacked2DCNNConfig, ViTConfig
from ludwig.utils.torch_utils import FreezeModule

logger = logging.getLogger(__name__)


@DeveloperAPI
class ImageEncoder(Encoder):
    pass


@DeveloperAPI
@register_encoder("stacked_cnn", IMAGE)
class Stacked2DCNN(ImageEncoder):
    def __init__(
        self,
        height: int,
        width: int,
        conv_layers: Optional[List[Dict]] = None,
        num_conv_layers: Optional[int] = None,
        num_channels: int = None,
        out_channels: int = 32,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int], str] = "valid",
        dilation: Union[int, Tuple[int]] = 1,
        conv_use_bias: bool = True,
        padding_mode: str = "zeros",
        conv_norm: Optional[str] = None,
        conv_norm_params: Optional[Dict[str, Any]] = None,
        conv_activation: str = "relu",
        conv_dropout: int = 0,
        pool_function: str = "max",
        pool_kernel_size: Union[int, Tuple[int]] = 2,
        pool_stride: Union[int, Tuple[int]] = None,
        pool_padding: Union[int, Tuple[int]] = 0,
        pool_dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        fc_layers: Optional[List[Dict]] = None,
        num_fc_layers: Optional[int] = 1,
        output_size: int = 128,
        fc_use_bias: bool = True,
        fc_weights_initializer: str = "xavier_uniform",
        fc_bias_initializer: str = "zeros",
        fc_norm: Optional[str] = None,
        fc_norm_params: Optional[Dict[str, Any]] = None,
        fc_activation: str = "relu",
        fc_dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        # map parameter input feature config names to internal names
        img_height = height
        img_width = width
        first_in_channels = num_channels

        self._input_shape = (first_in_channels, img_height, img_width)

        if first_in_channels is None:
            raise ValueError("first_in_channels must not be None.")

        logger.debug("  Conv2DStack")
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
            default_use_bias=conv_use_bias,
            default_padding_mode=padding_mode,
            default_norm=conv_norm,
            default_norm_params=conv_norm_params,
            default_activation=conv_activation,
            default_dropout=conv_dropout,
            default_pool_function=pool_function,
            default_pool_kernel_size=pool_kernel_size,
            default_pool_stride=pool_stride,
            default_pool_padding=pool_padding,
            default_pool_dilation=pool_dilation,
        )
        out_channels, img_height, img_width = self.conv_stack_2d.output_shape
        first_fc_layer_input_size = out_channels * img_height * img_width

        self.flatten = torch.nn.Flatten()

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_output_size=output_size,
            default_use_bias=fc_use_bias,
            default_weights_initializer=fc_weights_initializer,
            default_bias_initializer=fc_bias_initializer,
            default_norm=fc_norm,
            default_norm_params=fc_norm_params,
            default_activation=fc_activation,
            default_dropout=fc_dropout,
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
        return Stacked2DCNNConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


@DeveloperAPI
@register_encoder("_resnet_legacy", IMAGE)
class ResNetEncoder(ImageEncoder):
    def __init__(
        self,
        height: int,
        width: int,
        resnet_size: int = 50,
        num_channels: int = 3,
        out_channels: int = 16,
        kernel_size: Union[int, Tuple[int]] = 3,
        conv_stride: Union[int, Tuple[int]] = 1,
        first_pool_kernel_size: Union[int, Tuple[int]] = None,
        first_pool_stride: Union[int, Tuple[int]] = None,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 0.001,
        fc_layers: Optional[List[Dict]] = None,
        num_fc_layers: Optional[int] = 1,
        output_size: int = 256,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: Optional[str] = None,
        norm_params: Optional[Dict[str, Any]] = None,
        activation: str = "relu",
        dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")
        # map parameter input feature config names to internal names
        img_height = height
        img_width = width
        first_in_channels = num_channels

        self._input_shape = (first_in_channels, img_height, img_width)

        logger.debug("  ResNet")
        self.resnet = ResNet(
            img_height=img_height,
            img_width=img_width,
            first_in_channels=first_in_channels,
            out_channels=out_channels,
            resnet_size=resnet_size,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            first_pool_kernel_size=first_pool_kernel_size,
            first_pool_stride=first_pool_stride,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon,
        )
        first_fc_layer_input_size = self.resnet.output_shape[0]

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=first_fc_layer_input_size,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_output_size=output_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.resnet(inputs)
        axes = [2, 3]
        hidden = torch.mean(hidden, axes)
        hidden = self.fc_stack(hidden)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return ResNetConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


@DeveloperAPI
@register_encoder("mlp_mixer", IMAGE)
class MLPMixerEncoder(ImageEncoder):
    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int = None,
        patch_size: int = 16,
        embed_size: int = 512,
        token_size: int = 2048,
        channel_dim: int = 256,
        num_layers: int = 8,
        dropout: float = 0.0,
        avg_pool: bool = True,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")
        # map parameter input feature config names to internal names
        img_height = height
        img_width = width
        in_channels = num_channels

        if num_channels is None:
            raise RuntimeError("num_channels must not be None")

        self._input_shape = (in_channels, img_height, img_width)

        logger.debug("  MLPMixer")
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

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.mlp_mixer(inputs)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return MLPMixerConfig

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape


@DeveloperAPI
@register_encoder("_vit_legacy", IMAGE)
class ViTEncoder(ImageEncoder):
    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int = 3,
        use_pretrained: bool = True,
        pretrained_model: str = "google/vit-base-patch16-224",
        saved_weights_in_checkpoint: bool = False,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        gradient_checkpointing: bool = False,
        patch_size: int = 16,
        trainable: bool = True,
        output_attentions: bool = False,
        encoder_config=None,
        **kwargs,
    ):
        """Creates a ViT encoder using transformers.ViTModel.

        use_pretrained: If True, uses a pretrained transformer based on the
            pretrained_model argument.
        pretrained: If str, expects the path to a pretrained model or the id of
            a model on huggingface.co, and ignores the configuration provided in
            the arguments.
        """
        super().__init__()
        self.config = encoder_config

        try:
            from transformers import ViTConfig, ViTModel
        except ModuleNotFoundError:
            raise RuntimeError(
                " transformers is not installed. "
                "In order to install all image feature dependencies run "
                "pip install ludwig[image]"
            )

        # map parameter input feature config names to internal names
        img_height = height
        img_width = width
        in_channels = num_channels

        img_width = img_width or img_height
        if img_width != img_height:
            raise ValueError("img_height and img_width should be identical.")
        self._input_shape = (in_channels, img_height, img_width)

        if use_pretrained and not saved_weights_in_checkpoint:
            transformer = ViTModel.from_pretrained(pretrained_model)
        else:
            config = ViTConfig(
                image_size=img_height,
                num_channels=in_channels,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                gradient_checkpointing=gradient_checkpointing,
            )
            transformer = ViTModel(config)

        self.transformer = FreezeModule(transformer, frozen=not trainable)

        self._output_shape = (transformer.config.hidden_size,)
        self.output_attentions = output_attentions

    def forward(self, inputs: torch.Tensor, head_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        output = self.transformer.module(inputs, head_mask=head_mask, output_attentions=self.output_attentions)
        return_dict = {"encoder_output": output.pooler_output}
        if self.output_attentions:
            return_dict["attentions"] = output.attentions
        return return_dict

    @staticmethod
    def get_schema_cls():
        return ViTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)
