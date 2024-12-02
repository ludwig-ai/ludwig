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
from typing import Dict, Optional, Type

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT_STATE, HIDDEN, IMAGE, LOGITS, PREDICTIONS
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.modules.convolutional_modules import UNetUpStack
from ludwig.schema.decoders.image_decoders import ImageDecoderConfig, UNetDecoderConfig

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_decoder("unet", IMAGE)
class UNetDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        height: int,
        width: int,
        num_channels: int = 1,
        num_classes: int = 2,
        conv_norm: Optional[str] = None,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.num_classes = num_classes

        logger.debug(f" {self.name}")
        if num_classes < 2:
            raise ValueError(f"Invalid `num_classes` {num_classes} for unet decoder")
        if height % 16 or width % 16:
            raise ValueError(f"Invalid `height` {height} or `width` {width} for unet decoder")

        self.unet = UNetUpStack(
            img_height=height,
            img_width=width,
            out_channels=num_classes,
            norm=conv_norm,
        )

        self.input_reshape = list(self.unet.input_shape)
        self.input_reshape.insert(0, -1)
        self._output_shape = (height, width)

    def forward(self, combiner_outputs: Dict[str, torch.Tensor], target: torch.Tensor):
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
    def get_schema_cls() -> Type[ImageDecoderConfig]:
        return UNetDecoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)

    @property
    def input_shape(self) -> torch.Size:
        return self.unet.input_shape
