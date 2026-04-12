#! /usr/bin/env python
# Copyright (c) 2023 Aizen Corp.
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT_STATE, HIDDEN, IMAGE, LOGITS, PREDICTIONS
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.modules.convolutional_modules import UNetUpStack
from ludwig.schema.decoders.image_decoders import (
    FPNDecoderConfig,
    ImageDecoderConfig,
    SegFormerDecoderConfig,
    UNetDecoderConfig,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_decoder("unet", IMAGE)
class UNetDecoder(Decoder):
    """U-Net decoder for dense pixel-level prediction (e.g. semantic segmentation).

    Implements the expansive upsampling path of U-Net, consisting of ``num_stages``
    transposed-convolution up-sampling blocks with skip connections from the encoder.
    Each block doubles the spatial resolution and halves the channel count, ending
    with a 1×1 convolution that maps to ``num_classes`` output channels.

    Choose this decoder when:
    - Your encoder also follows a U-Net / encoder-decoder structure and provides skip
      connections via ``ENCODER_OUTPUT_STATE``.
    - You need high-resolution, pixel-accurate segmentation masks.
    - You want a well-understood, battle-tested architecture.

    Reference:
        Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
        Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        input_size: int,
        height: int,
        width: int,
        num_channels: int = 1,
        num_classes: int = 2,
        conv_norm: str | None = None,
        num_stages: int = 4,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.num_classes = num_classes

        logger.debug(f" {self.name}")
        if num_classes < 2:
            raise ValueError(f"Invalid `num_classes` {num_classes} for unet decoder")

        divisor = 2**num_stages
        if height % divisor or width % divisor:
            raise ValueError(
                f"Invalid `height` {height} or `width` {width} for unet decoder with "
                f"num_stages={num_stages}: dimensions must be divisible by {divisor}"
            )

        self.unet = UNetUpStack(
            img_height=height,
            img_width=width,
            out_channels=num_classes,
            norm=conv_norm,
            stack_depth=num_stages,
        )

        self.input_reshape = list(self.unet.input_shape)
        self.input_reshape.insert(0, -1)
        self._output_shape = (height, width)

    def forward(self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor):
        hidden = combiner_outputs[HIDDEN]
        skips = combiner_outputs[ENCODER_OUTPUT_STATE]

        # unflatten combiner outputs
        hidden = hidden.reshape(self.input_reshape)

        logits = self.unet(hidden, skips)
        predictions = logits.argmax(dim=1).squeeze(1).byte()

        return {LOGITS: logits, PREDICTIONS: predictions}

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS}

    @staticmethod
    def get_schema_cls() -> type[ImageDecoderConfig]:
        return UNetDecoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return self.unet.input_shape


@DeveloperAPI
@register_decoder("segformer", IMAGE)
class SegFormerDecoder(Decoder):
    """Lightweight all-MLP decoder head for semantic segmentation.

    Takes the flat feature vector produced by the combiner, reshapes it into a
    spatial feature map (using the square-root of ``input_size // hidden_size``
    as the intermediate spatial extent), applies a two-layer MLP projection, then
    bilinearly upsamples to the target ``(height, width)`` and produces per-pixel
    class logits with a final 1×1 convolution.

    This design is inspired by the SegFormer decode head which intentionally omits
    complex spatial attention so that the encoder (typically a hierarchical
    transformer) carries the representational burden.

    Choose this decoder when:
    - Your encoder is a transformer (ViT, Swin, DeiT, …) and produces a rich,
      globally-aware feature vector.
    - You want a fast, low-parameter decoder that does not bottleneck training.
    - Memory is constrained — no skip connections or transposed convolutions.

    Reference:
        Xie et al., "SegFormer: Simple and Efficient Design for Semantic
        Segmentation with Transformers", NeurIPS 2021.
        https://arxiv.org/abs/2105.15203
    """

    def __init__(
        self,
        input_size: int,
        height: int,
        width: int,
        num_channels: int = 1,
        num_classes: int = 2,
        hidden_size: int = 256,
        dropout: float = 0.1,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.num_classes = num_classes
        self.height = height
        self.width = width

        logger.debug(f" {self.name}")
        if num_classes < 2:
            raise ValueError(f"Invalid `num_classes` {num_classes} for segformer decoder")

        # MLP projection: input_size → hidden_size → hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        # Final 1×1 conv to produce per-pixel class scores.
        # We treat the hidden_size channels as a (hidden_size, 1, 1) feature map
        # and upsample to (height, width) before applying this conv.
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

        self._input_shape = torch.Size([input_size])
        self._output_shape = torch.Size([height, width])

    def forward(self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor):
        hidden = combiner_outputs[HIDDEN]  # (B, input_size)

        # MLP projection
        features = self.mlp(hidden)  # (B, hidden_size)

        # Reshape to (B, hidden_size, 1, 1) and upsample to target resolution
        features = features.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_size, 1, 1)
        features = F.interpolate(features, size=(self.height, self.width), mode="bilinear", align_corners=False)

        # Per-pixel classification
        logits = self.classifier(features)  # (B, num_classes, H, W)
        predictions = logits.argmax(dim=1).squeeze(1).byte()  # (B, H, W)

        return {LOGITS: logits, PREDICTIONS: predictions}

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS}

    @staticmethod
    def get_schema_cls() -> type[ImageDecoderConfig]:
        return SegFormerDecoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape


@DeveloperAPI
@register_decoder("fpn", IMAGE)
class FPNDecoder(Decoder):
    """Feature Pyramid Network (FPN) decoder for multi-scale segmentation.

    Builds a feature pyramid using lateral 1×1 projections and a top-down
    pathway.  The flat feature vector from the combiner is reshaped into a
    spatial map at the coarsest scale, then progressively upsampled and merged
    across ``num_levels`` levels.  Each level doubles the spatial resolution.
    All levels are upsampled to the finest scale, concatenated, and projected
    with a 3×3 convolution followed by a 1×1 classifier to produce per-pixel
    logits.

    Choose this decoder when:
    - Your task requires detecting / segmenting objects at multiple scales
      simultaneously.
    - You have a relatively powerful encoder (CNN or ViT) whose output you want
      to leverage at different resolutions.
    - You can afford slightly more compute than the SegFormer MLP head.

    Reference:
        Lin et al., "Feature Pyramid Networks for Object Detection",
        CVPR 2017. https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        input_size: int,
        height: int,
        width: int,
        num_classes: int = 2,
        num_channels: int = 256,
        num_levels: int = 4,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.height = height
        self.width = width

        logger.debug(f" {self.name}")
        if num_classes < 2:
            raise ValueError(f"Invalid `num_classes` {num_classes} for fpn decoder")
        if num_levels < 1:
            raise ValueError(f"Invalid `num_levels` {num_levels} for fpn decoder; must be >= 1")

        # Determine the coarsest spatial size (level 0 of the pyramid).
        # The coarsest height/width is height / 2^(num_levels-1), rounded up.
        self.coarse_h = math.ceil(height / (2 ** (num_levels - 1)))
        self.coarse_w = math.ceil(width / (2 ** (num_levels - 1)))

        # Project the flat combiner output into (num_channels, coarse_h, coarse_w)
        coarse_spatial = self.coarse_h * self.coarse_w
        self.input_proj = nn.Linear(input_size, num_channels * coarse_spatial)

        # Top-down lateral projections: one 1×1 conv per level (applied to
        # the up-sampled feature from the previous level).  Level 0 is the
        # coarsest and requires no lateral merge — we start from level 1.
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(num_channels, num_channels, kernel_size=1) for _ in range(num_levels - 1)]
        )

        # After merging all levels at the finest scale we apply a 3×3 conv
        # over all concatenated levels to mix information.
        self.merge_conv = nn.Conv2d(num_channels * num_levels, num_channels, kernel_size=3, padding=1)

        # Final 1×1 classifier
        self.classifier = nn.Conv2d(num_channels, num_classes, kernel_size=1)

        self._input_shape = torch.Size([input_size])
        self._output_shape = torch.Size([height, width])

    def forward(self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor):
        hidden = combiner_outputs[HIDDEN]  # (B, input_size)
        batch_size = hidden.shape[0]

        # Project and reshape to coarsest spatial feature map
        coarse = self.input_proj(hidden)  # (B, num_channels * coarse_h * coarse_w)
        coarse = coarse.view(batch_size, self.num_channels, self.coarse_h, self.coarse_w)

        # Build the top-down pyramid.
        # pyramid[0] is the coarsest; pyramid[-1] will be the finest.
        pyramid = [coarse]
        current = coarse
        for lateral_conv in self.lateral_convs:
            # Double the spatial resolution
            upsampled = F.interpolate(current, scale_factor=2, mode="nearest")
            current = lateral_conv(upsampled)
            pyramid.append(current)

        # Upsample all levels to the finest pyramid resolution, then to target size.
        finest_h = pyramid[-1].shape[2]
        finest_w = pyramid[-1].shape[3]

        merged = []
        for level_feat in pyramid:
            if level_feat.shape[2] != finest_h or level_feat.shape[3] != finest_w:
                level_feat = F.interpolate(level_feat, size=(finest_h, finest_w), mode="bilinear", align_corners=False)
            level_feat = F.interpolate(level_feat, size=(self.height, self.width), mode="bilinear", align_corners=False)
            merged.append(level_feat)

        # Concatenate across channel dimension and mix
        fused = torch.cat(merged, dim=1)  # (B, num_channels * num_levels, H, W)
        fused = self.merge_conv(fused)  # (B, num_channels, H, W)

        logits = self.classifier(fused)  # (B, num_classes, H, W)
        predictions = logits.argmax(dim=1).squeeze(1).byte()  # (B, H, W)

        return {LOGITS: logits, PREDICTIONS: predictions}

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS}

    @staticmethod
    def get_schema_cls() -> type[ImageDecoderConfig]:
        return FPNDecoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape

    @property
    def input_shape(self) -> torch.Size:
        return self._input_shape
