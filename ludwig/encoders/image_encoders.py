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
import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.models as tvm

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.convolutional_modules import Conv2DStack
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.mlp_mixer_modules import MLPMixer
from ludwig.schema.encoders.image_encoders import (
    MLPMixerEncoderConfig,
    Stacked2DCNNEncoderConfig,
    TVAlexNetEncoderConfig,
    TVConvNeXtEncoderConfig,
    TVDenseNetEncoderConfig,
    TVEfficientNetEncoderConfig,
    TVGoogLeNetEncoderConfig,
    TVInceptionV3EncoderConfig,
    TVMaxVitEncoderConfig,
    TVMNASNetEncoderConfig,
    TVMobileNetV2EncoderConfig,
    TVMobileNetV3EncoderConfig,
    TVRegNetEncoderConfig,
    TVResNetEncoderConfig,
    TVResNeXtEncoderConfig,
    TVShuffleNetV2EncoderConfig,
    TVSqueezeNetEncoderConfig,
    TVSwinTransformerEncoderConfig,
    TVVGGEncoderConfig,
    TVViTEncoderConfig,
    TVWideResNetEncoderConfig,
)
from ludwig.utils.image_utils import register_torchvision_model_variants, torchvision_model_registry, TVModelVariant

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("stacked_cnn", IMAGE)
class Stacked2DCNN(Encoder):
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
        return Stacked2DCNNEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)


@register_encoder("mlp_mixer", IMAGE)
class MLPMixerEncoder(Encoder):
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
        return MLPMixerEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape


# TODO: Temporarily comment out, may be re-enabled later date as HF encoder
# @DeveloperAPI
# @register_encoder("vit", IMAGE)
# class ViTEncoder(Encoder):
#     def __init__(
#         self,
#         height: int,
#         width: int,
#         num_channels: int = 3,
#         use_pretrained: bool = True,
#         pretrained_model: str = "google/vit-base-patch16-224",
#         saved_weights_in_checkpoint: bool = False,
#         hidden_size: int = 768,
#         num_hidden_layers: int = 12,
#         num_attention_heads: int = 12,
#         intermediate_size: int = 3072,
#         hidden_act: str = "gelu",
#         hidden_dropout_prob: float = 0.1,
#         attention_probs_dropout_prob: float = 0.1,
#         initializer_range: float = 0.02,
#         layer_norm_eps: float = 1e-12,
#         gradient_checkpointing: bool = False,
#         patch_size: int = 16,
#         trainable: bool = True,
#         output_attentions: bool = False,
#         encoder_config=None,
#         **kwargs,
#     ):
#         """Creates a ViT encoder using transformers.ViTModel.
#
#         use_pretrained: If True, uses a pretrained transformer based on the
#             pretrained_model argument.
#         pretrained: If str, expects the path to a pretrained model or the id of
#             a model on huggingface.co, and ignores the configuration provided in
#             the arguments.
#         """
#         super().__init__()
#         self.config = encoder_config
#
#         try:
#             from transformers import ViTConfig, ViTModel
#         except ModuleNotFoundError:
#             raise RuntimeError(
#                 " transformers is not installed. "
#                 "In order to install all image feature dependencies run "
#                 "pip install ludwig[image]"
#             )
#
#         # map parameter input feature config names to internal names
#         img_height = height
#         img_width = width
#         in_channels = num_channels
#
#         img_width = img_width or img_height
#         if img_width != img_height:
#             raise ValueError("img_height and img_width should be identical.")
#         self._input_shape = (in_channels, img_height, img_width)
#
#         if use_pretrained and not saved_weights_in_checkpoint:
#             self.transformer = ViTModel.from_pretrained(pretrained_model)
#         else:
#             config = ViTConfig(
#                 image_size=img_height,
#                 num_channels=in_channels,
#                 patch_size=patch_size,
#                 hidden_size=hidden_size,
#                 num_hidden_layers=num_hidden_layers,
#                 num_attention_heads=num_attention_heads,
#                 intermediate_size=intermediate_size,
#                 hidden_act=hidden_act,
#                 hidden_dropout_prob=hidden_dropout_prob,
#                 attention_probs_dropout_prob=attention_probs_dropout_prob,
#                 initializer_range=initializer_range,
#                 layer_norm_eps=layer_norm_eps,
#                 gradient_checkpointing=gradient_checkpointing,
#             )
#             self.transformer = ViTModel(config)
#
#         if trainable:
#             self.transformer.train()
#         else:
#             freeze_parameters(self.transformer)
#
#         self._output_shape = (self.transformer.config.hidden_size,)
#         self.output_attentions = output_attentions
#
#     def forward(self, inputs: torch.Tensor, head_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
#         output = self.transformer(inputs, head_mask=head_mask, output_attentions=self.output_attentions)
#         return_dict = {"encoder_output": output.pooler_output}
#         if self.output_attentions:
#             return_dict["attentions"] = output.attentions
#         return return_dict
#
#     @staticmethod
#     def get_schema_cls():
#         return ViTEncoderConfig
#
#     @property
#     def input_shape(self) -> torch.Size:
#         return torch.Size(self._input_shape)
#
#     @property
#     def output_shape(self) -> torch.Size:
#         return torch.Size(self._output_shape)


class TVBaseEncoder(Encoder):
    def __init__(
        self,
        model_variant: Union[str, int] = None,
        use_pretrained: bool = True,
        saved_weights_in_checkpoint: bool = False,
        model_cache_dir: Optional[str] = None,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__()

        logger.debug(f" {self.name}")
        # map parameter input feature config names to internal names
        self.model_variant = model_variant
        self.use_pretrained = use_pretrained
        self.model_cache_dir = model_cache_dir

        # remove any Ludwig specific keyword parameters
        kwargs.pop("encoder_config", None)
        kwargs.pop("type", None)

        # cache pre-trained models if requested
        # based on https://github.com/pytorch/vision/issues/616#issuecomment-428637564
        if self.model_cache_dir is not None:
            os.environ["TORCH_HOME"] = self.model_cache_dir

        # model_id = f"{self.torchvision_model_type}-{self.model_variant}"

        # retrieve function to create requested model
        self.create_model = (
            torchvision_model_registry[self.torchvision_model_type][self.model_variant].create_model_function
        )

        # get weight specification
        if use_pretrained and not saved_weights_in_checkpoint:
            weights_specification = (
                torchvision_model_registry[self.torchvision_model_type][self.model_variant].model_weights.DEFAULT
            )
            logger.info(
                f"Instantiating torchvision image encoder '{self.torchvision_model_type}' with pretrained weights: "
                f"{weights_specification}."
            )
        else:
            weights_specification = None
            if saved_weights_in_checkpoint:
                logger.info(
                    f"Instantiating torchvision image encoder: '{self.torchvision_model_type}' "
                    "with weights saved in the checkpoint."
                )
            else:
                logger.info(
                    f"Instantiating torchvision image encoder: '{self.torchvision_model_type}' "
                    "with no pretrained weights."
                )

        # get torchvision transforms object
        transforms_obj = (
            torchvision_model_registry[self.torchvision_model_type][
                self.model_variant].model_weights.DEFAULT.transforms()
        )
        self.num_channels = len(transforms_obj.mean)
        self.crop_size = transforms_obj.crop_size

        logger.debug(f"  {self.torchvision_model_type}")
        # create pretrained model with pretrained weights or None for untrained model
        self.model = self.create_model(weights=weights_specification, **kwargs)

        # remove final classification layer
        self._remove_softmax_layer()

        # freeze parameters if requested
        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"encoder_output": self.model(inputs)}

    @abstractmethod
    def _remove_softmax_layer(self):
        """Model specific method that allows the final softmax layer to be implemented in the Ludwig Decoder
        component.  The model specific implementation should change the final softmax layer in the torchvision
        model architecture to torch.nn.Identity().  This allows the output tensor from the preceding layer to be
        passed to the Ludwig Combiner and then to the Decoder.

        Returns: None
        """
        raise NotImplementedError()

    @property
    def output_shape(self) -> torch.Size:
        # create synthetic image and run through forward method
        inputs = torch.randn([1, *self.input_shape])
        output = self.model(inputs)
        return torch.Size(output.shape[1:])

    @property
    def input_shape(self) -> torch.Size:
        # expected shape after all pre-processing
        # len(transforms_obj.mean) determines the number of channels
        # transforms_obj.crop_size determines the height and width of image
        # [num_channels, height, width]
        return torch.Size([self.num_channels, *(2 * self.crop_size)])


# TVModelVariant(variant_id, create_model_function, model_weights)
#   variant_id: model variant identifier
#   create_model_function: TorchVision function to create model class
#   model_weights: Torchvision class for model weights

ALEXNET_VARIANTS = [
    TVModelVariant("base", tvm.alexnet, tvm.AlexNet_Weights),
]


@register_torchvision_model_variants([
    TVModelVariant(variant_id="base", create_model_function=tvm.alexnet, model_weights=tvm.AlexNet_Weights),
])
@register_encoder("alexnet_torch", IMAGE)
class TVAlexNetEncoder(TVBaseEncoder):
    # specify base model type
    torchvision_model_type: str = "alexnet_torch"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    # TODO: discussion w/ justin
    # @property
    # def get_torchvision_model_type(self):
    #     return "alexnet_torch"

    def _remove_softmax_layer(self):
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls():
        return TVAlexNetEncoderConfig


# CONVNEXT_VARIANTS = [
#     TVModelVariant("tiny", tvm.convnext_tiny, tvm.convnext.ConvNeXt_Tiny_Weights),
#     TVModelVariant("small", tvm.convnext_small, tvm.convnext.ConvNeXt_Small_Weights),
#     TVModelVariant("base", tvm.convnext_base, tvm.ConvNeXt_Base_Weights),
#     TVModelVariant("large", tvm.convnext_large, tvm.ConvNeXt_Large_Weights),
# ]
#
#
# @register_torchvision_variants(CONVNEXT_VARIANTS)
# @register_encoder("convnext_torch", IMAGE)
# class TVConvNeXtEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "convnext_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVConvNeXtEncoderConfig
#
#
# DENSENET_VARIANTS = [
#     TVModelVariant(121, tvm.densenet121, tvm.DenseNet121_Weights),
#     TVModelVariant(161, tvm.densenet161, tvm.DenseNet161_Weights),
#     TVModelVariant(169, tvm.densenet169, tvm.DenseNet169_Weights),
#     TVModelVariant(201, tvm.densenet201, tvm.DenseNet201_Weights),
# ]
#
#
# @register_torchvision_variants(DENSENET_VARIANTS)
# @register_encoder("densenet_torch", IMAGE)
# class TVDenseNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "densenet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVDenseNetEncoderConfig
#
#
# EFFICIENTNET_VARIANTS = [
#     TVModelVariant("b0", tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights),
#     TVModelVariant("b1", tvm.efficientnet_b1, tvm.EfficientNet_B1_Weights),
#     TVModelVariant("b2", tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights),
#     TVModelVariant("b3", tvm.efficientnet_b3, tvm.EfficientNet_B3_Weights),
#     TVModelVariant("b4", tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights),
#     TVModelVariant("b5", tvm.efficientnet_b5, tvm.EfficientNet_B5_Weights),
#     TVModelVariant("b6", tvm.efficientnet_b6, tvm.EfficientNet_B6_Weights),
#     TVModelVariant("b7", tvm.efficientnet_b7, tvm.EfficientNet_B7_Weights),
#     TVModelVariant("v2_s", tvm.efficientnet_v2_s, tvm.EfficientNet_V2_S_Weights),
#     TVModelVariant("v2_m", tvm.efficientnet_v2_m, tvm.EfficientNet_V2_M_Weights),
#     TVModelVariant("v2_l", tvm.efficientnet_v2_l, tvm.EfficientNet_V2_L_Weights),
# ]
#
#
# @register_torchvision_variants(EFFICIENTNET_VARIANTS)
# @register_encoder("efficientnet_torch", IMAGE)
# class TVEfficientNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "efficientnet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVEfficientNetEncoderConfig
#
#
# GOOGLENET_VARIANTS = [
#     TVModelVariant("base", tvm.googlenet, tvm.GoogLeNet_Weights),
# ]
#
#
# @register_torchvision_variants(GOOGLENET_VARIANTS)
# @register_encoder("googlenet_torch", IMAGE)
# class TVGoogLeNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "googlenet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#         # if auxiliary network exists, eliminate auxiliary network
#         # to resolve issue when loading a saved model which does not
#         # contain the auxiliary network
#         if self.model.aux_logits:
#             self.model.aux_logits = False
#             self.model.aux1 = None
#             self.model.aux2 = None
#
#     def _remove_softmax_layer(self):
#         self.model.fc = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVGoogLeNetEncoderConfig
#
#
# INCEPTIONV3_VARIANTS = [
#     TVModelVariant("base", tvm.inception_v3, tvm.Inception_V3_Weights),
# ]
#
#
# @register_torchvision_variants(INCEPTIONV3_VARIANTS)
# @register_encoder("inceptionv3_torch", IMAGE)
# class TVInceptionV3Encoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "inceptionv3_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#         # if auxiliary network exists, eliminate auxiliary network
#         # to resolve issue when loading a saved model which does not
#         # contain the auxiliary network
#         if self.model.aux_logits:
#             self.model.aux_logits = False
#             self.model.AuxLogits = None
#
#     def _remove_softmax_layer(self):
#         self.model.fc = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVInceptionV3EncoderConfig
#
#
# MAXVIT_VARIANTS = [
#     TVModelVariant("t", tvm.maxvit_t, tvm.MaxVit_T_Weights),
# ]
#
#
# @register_torchvision_variants(MAXVIT_VARIANTS)
# @register_encoder("maxvit_torch", IMAGE)
# class TVMaxVitEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "maxvit_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVMaxVitEncoderConfig
#
#
# MNASNET_VARIANTS = [
#     TVModelVariant("0_5", tvm.mnasnet0_5, tvm.mnasnet.MNASNet0_5_Weights),
#     TVModelVariant("0_75", tvm.mnasnet0_75, tvm.mnasnet.MNASNet0_75_Weights),
#     TVModelVariant("1_0", tvm.mnasnet1_0, tvm.mnasnet.MNASNet1_0_Weights),
#     TVModelVariant("1_3", tvm.mnasnet1_3, tvm.mnasnet.MNASNet1_3_Weights),
# ]
#
#
# @register_torchvision_variants(MNASNET_VARIANTS)
# @register_encoder("mnasnet_torch", IMAGE)
# class TVMNASNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "mnasnet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVMNASNetEncoderConfig
#
#
# MOBILENETV2_VARIANTS = [
#     TVModelVariant("base", tvm.mobilenet_v2, tvm.MobileNet_V2_Weights),
# ]
#
#
# @register_torchvision_variants(MOBILENETV2_VARIANTS)
# @register_encoder("mobilenetv2_torch", IMAGE)
# class TVMobileNetV2Encoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "mobilenetv2_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVMobileNetV2EncoderConfig
#
#
# MOBILENETV3_VARIANTS = [
#     TVModelVariant("small", tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights),
#     TVModelVariant("large", tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights),
# ]
#
#
# @register_torchvision_variants(MOBILENETV3_VARIANTS)
# @register_encoder("mobilenetv3_torch", IMAGE)
# class TVMobileNetV3Encoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "mobilenetv3_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVMobileNetV3EncoderConfig
#
#
# REGNET_VARIANTS = [
#     TVModelVariant("x_16gf", tvm.regnet_x_16gf, tvm.RegNet_X_16GF_Weights),
#     TVModelVariant("x_1_6gf", tvm.regnet_x_1_6gf, tvm.RegNet_X_1_6GF_Weights),
#     TVModelVariant("x_32gf", tvm.regnet_x_32gf, tvm.RegNet_X_32GF_Weights),
#     TVModelVariant("x_3_2gf", tvm.regnet_x_3_2gf, tvm.RegNet_X_3_2GF_Weights),
#     TVModelVariant("x_400mf", tvm.regnet_x_400mf, tvm.RegNet_X_400MF_Weights),
#     TVModelVariant("x_800mf", tvm.regnet_x_800mf, tvm.RegNet_X_800MF_Weights),
#     TVModelVariant("x_8gf", tvm.regnet_x_8gf, tvm.RegNet_X_8GF_Weights),
#     TVModelVariant("y_128gf", tvm.regnet_y_128gf, tvm.RegNet_Y_128GF_Weights),
#     TVModelVariant("y_16gf", tvm.regnet_y_16gf, tvm.RegNet_Y_16GF_Weights),
#     TVModelVariant("y_1_6gf", tvm.regnet_y_1_6gf, tvm.RegNet_Y_1_6GF_Weights),
#     TVModelVariant("y_32gf", tvm.regnet_y_32gf, tvm.RegNet_Y_32GF_Weights),
#     TVModelVariant("y_3_2gf", tvm.regnet_y_3_2gf, tvm.RegNet_Y_3_2GF_Weights),
#     TVModelVariant("y_400mf", tvm.regnet_y_400mf, tvm.RegNet_Y_400MF_Weights),
#     TVModelVariant("y_800mf", tvm.regnet_y_800mf, tvm.RegNet_Y_800MF_Weights),
#     TVModelVariant("y_8gf", tvm.regnet_y_8gf, tvm.RegNet_Y_8GF_Weights),
# ]
#
#
# @register_torchvision_variants(REGNET_VARIANTS)
# @register_encoder("regnet_torch", IMAGE)
# class TVRegNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "regnet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.fc = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVRegNetEncoderConfig
#
#
# RESNET_TORCH_VARIANTS = [
#     TVModelVariant(18, tvm.resnet18, tvm.ResNet18_Weights),
#     TVModelVariant(34, tvm.resnet34, tvm.ResNet34_Weights),
#     TVModelVariant(50, tvm.resnet50, tvm.ResNet50_Weights),
#     TVModelVariant(101, tvm.resnet101, tvm.ResNet101_Weights),
#     TVModelVariant(152, tvm.resnet152, tvm.ResNet152_Weights),
# ]
# @register_torchvision_model_variants([
#     TVModelVariant(variant_id="base", create_model_function=tvm.alexnet, model_weights=tvm.AlexNet_Weights),
# ])

#
#
@register_torchvision_model_variants([
    TVModelVariant(18, tvm.resnet18, tvm.ResNet18_Weights),
    TVModelVariant(34, tvm.resnet34, tvm.ResNet34_Weights),
    TVModelVariant(50, tvm.resnet50, tvm.ResNet50_Weights),
    TVModelVariant(101, tvm.resnet101, tvm.ResNet101_Weights),
    TVModelVariant(152, tvm.resnet152, tvm.ResNet152_Weights),
])
@register_encoder("resnet_torch", IMAGE)
class TVResNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "resnet_torch"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self):
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls():
        return TVResNetEncoderConfig
#
#
# RESNEXT_VARIANTS = [
#     TVModelVariant("50_32x4d", tvm.resnext50_32x4d, tvm.ResNeXt50_32X4D_Weights),
#     TVModelVariant("101_328xd", tvm.resnext101_32x8d, tvm.ResNeXt101_32X8D_Weights),
#     TVModelVariant("101_64x4d", tvm.resnext101_64x4d, tvm.ResNeXt101_64X4D_Weights),
# ]
#
#
# @register_torchvision_variants(RESNEXT_VARIANTS)
# @register_encoder("resnext_torch", IMAGE)
# class TVResNeXtEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "resnext_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.fc = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVResNeXtEncoderConfig
#
#
# SHUFFLENET_V2_VARIANTS = [
#     TVModelVariant("x0_5", tvm.shufflenet_v2_x0_5, tvm.ShuffleNet_V2_X0_5_Weights),
#     TVModelVariant("x1_0", tvm.shufflenet_v2_x1_0, tvm.ShuffleNet_V2_X1_0_Weights),
#     TVModelVariant("x1_5", tvm.shufflenet_v2_x1_5, tvm.ShuffleNet_V2_X1_5_Weights),
#     TVModelVariant("x2_0", tvm.shufflenet_v2_x2_0, tvm.ShuffleNet_V2_X2_0_Weights),
# ]
#
#
# @register_torchvision_variants(SHUFFLENET_V2_VARIANTS)
# @register_encoder("shufflenet_v2_torch", IMAGE)
# class TVShuffleNetV2Encoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "shufflenet_v2_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.fc = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVShuffleNetV2EncoderConfig
#
#
# SQUEEZENET_VARIANTS = [
#     TVModelVariant("1_0", tvm.squeezenet1_0, tvm.SqueezeNet1_0_Weights),
#     TVModelVariant("1_1", tvm.squeezenet1_1, tvm.SqueezeNet1_1_Weights),
# ]
#
#
# @register_torchvision_variants(SQUEEZENET_VARIANTS)
# @register_encoder("squeezenet_torch", IMAGE)
# class TVSqueezeNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "squeezenet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         # SqueezeNet does not have a final nn.Linear() layer
#         # Use flatten output from last AdaptiveAvgPool2d layer
#         # as encoder output.
#         pass
#
#     @staticmethod
#     def get_schema_cls():
#         return TVSqueezeNetEncoderConfig
#
#
# SWIN_TRANSFORMER_VARIANTS = [
#     TVModelVariant("t", tvm.swin_t, tvm.Swin_T_Weights),
#     TVModelVariant("s", tvm.swin_s, tvm.Swin_S_Weights),
#     TVModelVariant("b", tvm.swin_b, tvm.Swin_B_Weights),
# ]
#
#
# @register_torchvision_variants(SWIN_TRANSFORMER_VARIANTS)
# @register_encoder("swin_transformer_torch", IMAGE)
# class TVSwinTransformerEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "swin_transformer_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.head = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVSwinTransformerEncoderConfig
#
#
# VGG_VARIANTS = [
#     TVModelVariant(11, tvm.vgg11, tvm.VGG11_Weights),
#     TVModelVariant("11_bn", tvm.vgg11_bn, tvm.VGG11_BN_Weights),
#     TVModelVariant(13, tvm.vgg13, tvm.VGG13_Weights),
#     TVModelVariant("13_bn", tvm.vgg13_bn, tvm.VGG13_BN_Weights),
#     TVModelVariant(16, tvm.vgg16, tvm.VGG16_Weights),
#     TVModelVariant("16_bn", tvm.vgg16_bn, tvm.VGG16_BN_Weights),
#     TVModelVariant(19, tvm.vgg19, tvm.VGG19_Weights),
#     TVModelVariant("19_bn", tvm.vgg19_bn, tvm.VGG19_BN_Weights),
# ]
#
#
# @register_torchvision_variants(VGG_VARIANTS)
# @register_encoder("vgg_torch", IMAGE)
# class TVVGGEncoder(TVBaseEncoder):
#     # specify base torchvison model
#     torchvision_model_type: str = "vgg_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.classifier[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVVGGEncoderConfig
#
#
# VIT_VARIANTS = [
#     TVModelVariant("b_16", tvm.vit_b_16, tvm.ViT_B_16_Weights),
#     TVModelVariant("b_32", tvm.vit_b_32, tvm.ViT_B_32_Weights),
#     TVModelVariant("l_16", tvm.vit_l_16, tvm.ViT_L_16_Weights),
#     TVModelVariant("l_32", tvm.vit_l_32, tvm.ViT_L_32_Weights),
#     TVModelVariant("h_14", tvm.vit_h_14, tvm.ViT_H_14_Weights),
# ]
#
#
# @register_torchvision_variants(VIT_VARIANTS)
# @register_encoder("vit_torch", IMAGE)
# class TVViTEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "vit_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#
#         # Depending on model variant and weight specification, the expected image size
#         # will vary.  This code determines at run time what the expected image size will be
#         # and adds to the kwargs dictionary the parameter that specifies the image size.
#         # this is needed only if not using pretrained weights.  If pre-trained weights are
#         # specified, then the correct image size is set.
#         if not kwargs["use_pretrained"]:
#             model_id = f"{self.torchvision_model_type}-{kwargs.get('model_variant')}"
#             weights_specification = torchvision_model_registry[model_id].weights_class.DEFAULT
#             kwargs["image_size"] = weights_specification.transforms.keywords["crop_size"]
#
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.heads[-1] = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVViTEncoderConfig
#
#
# WIDE_RESNET_VARIANTS = [
#     TVModelVariant("50_2", tvm.wide_resnet50_2, tvm.Wide_ResNet50_2_Weights),
#     TVModelVariant("101_2", tvm.wide_resnet101_2, tvm.Wide_ResNet101_2_Weights),
# ]
#
#
# @register_torchvision_variants(WIDE_RESNET_VARIANTS)
# @register_encoder("wide_resnet_torch", IMAGE)
# class TVWideResNetEncoder(TVBaseEncoder):
#     # specify base torchvision model
#     torchvision_model_type: str = "wide_resnet_torch"
#
#     def __init__(
#         self,
#         **kwargs,
#     ):
#         logger.debug(f" {self.name}")
#         super().__init__(**kwargs)
#
#     def _remove_softmax_layer(self):
#         self.model.fc = torch.nn.Identity()
#
#     @staticmethod
#     def get_schema_cls():
#         return TVWideResNetEncoderConfig
