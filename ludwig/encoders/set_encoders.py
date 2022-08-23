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

from ludwig.constants import SET
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import EmbedSet
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.set_encoders import SetSparseEncoderConfig

logger = logging.getLogger(__name__)


@register_encoder("embed", SET)
class SetSparseEncoder(Encoder):
    def __init__(self, encoder_config: SetSparseEncoderConfig = SetSparseEncoderConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")

        self.vocab_size = len(encoder_config.vocab)

        logger.debug("  Embed")
        self.embed = EmbedSet(
            encoder_config.vocab,
            encoder_config.embedding_size,
            representation=encoder_config.representation,
            embeddings_trainable=encoder_config.embeddings_trainable,
            pretrained_embeddings=encoder_config.pretrained_embeddings,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  FCStack")
        # TODO(shreya): Make sure this is updated when FCStack is updated
        self.fc_stack = FCStack(
            first_layer_input_size=self.embed.output_shape[-1],
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Params:
            inputs: The inputs fed into the encoder.
                    Shape: [batch x vocab_size], type tf.int32.

        Returns:
            Embeddings of shape [batch x vocab_size x embed size], type float32.
        """
        hidden = self.embed(inputs)
        hidden = self.fc_stack(hidden)

        return hidden

    @staticmethod
    def get_schema_cls():
        return SetSparseEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.vocab_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape
