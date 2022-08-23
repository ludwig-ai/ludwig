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

import torch

from ludwig.constants import BINARY, CATEGORY, NUMBER, VECTOR
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.base import DenseEncoderConfig, PassthroughEncoderConfig

logger = logging.getLogger(__name__)


@register_encoder("passthrough", [CATEGORY, NUMBER, VECTOR])
class PassthroughEncoder(Encoder):
    def __init__(self, encoder_config: PassthroughEncoderConfig = PassthroughEncoderConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")
        self.input_size = encoder_config.input_size

    def forward(self, inputs, mask=None):
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type tf.float32
        """
        return {"encoder_output": inputs}

    @staticmethod
    def get_schema_cls():
        return PassthroughEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape


@register_encoder("dense", [BINARY, NUMBER, VECTOR])
class DenseEncoder(Encoder):
    def __init__(self, encoder_config: DenseEncoderConfig = DenseEncoderConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")
        self.input_size = encoder_config.input_size

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=encoder_config.input_size,
            layers=encoder_config.layers,
            num_layers=encoder_config.num_layers,
            default_output_size=encoder_config.output_size,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=encoder_config.norm,
            default_norm_params=encoder_config.norm_params,
            default_activation=encoder_config.activation,
            default_dropout=encoder_config.dropout,
        )

    def forward(self, inputs, training=None, mask=None):
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type tf.float32
        """
        return {"encoder_output": self.fc_stack(inputs)}

    @staticmethod
    def get_schema_cls():
        return DenseEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.fc_stack.layers[-1]["output_size"]])
