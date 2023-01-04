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
from typing import Any, Dict, Set

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, MODEL_ECD, MODEL_GBM
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.schema.encoders.binary_encoders import BinaryPassthroughEncoderConfig

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("passthrough", BINARY)
class BinaryPassthroughEncoder(Encoder):
    def __init__(self, encoder_config=None, **kwargs):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type torch.float32
        """
        if inputs.dtype == torch.bool:
            inputs = inputs.to(torch.float32)

        return inputs

    @staticmethod
    def get_schema_cls():
        return BinaryPassthroughEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @classmethod
    def get_supported_model_types(cls, encoder_params: Dict[str, Any]) -> Set[str]:
        return {MODEL_ECD, MODEL_GBM}
