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

logger = logging.getLogger(__name__)


@register_encoder("passthrough", [CATEGORY, NUMBER, VECTOR], default=True)
class PassthroughEncoder(Encoder):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        logger.debug(f" {self.name}")
        self.input_size = input_size

    def forward(self, inputs, mask=None):
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type tf.float32
        """
        return {"encoder_output": inputs}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape


@register_encoder("dense", [BINARY, NUMBER, VECTOR])
class DenseEncoder(Encoder):
    def __init__(
        self,
        input_size,
        layers=None,
        num_layers=1,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        activation="relu",
        dropout=0,
        **kwargs,
    ):
        super().__init__()
        logger.debug(f" {self.name}")
        self.input_size = input_size

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=input_size,
            layers=layers,
            num_layers=num_layers,
            default_output_size=output_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
        )

    def forward(self, inputs, training=None, mask=None):
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type tf.float32
        """
        return {"encoder_output": self.fc_stack(inputs)}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.fc_stack.layers[-1]["output_size"]])
