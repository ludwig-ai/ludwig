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

from ludwig.constants import BAG
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import EmbedWeighted
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.bag_encoders import BagEmbedWeightedConfig

logger = logging.getLogger(__name__)


@register_encoder("embed", BAG)
class BagEmbedWeightedEncoder(Encoder):
    def __init__(self, encoder_config: BagEmbedWeightedConfig = BagEmbedWeightedConfig(), **kwargs):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")

        logger.debug("  EmbedWeighted")
        self.embed_weighted = EmbedWeighted(
            encoder_config.vocab,
            encoder_config.embedding_size,
            representation=encoder_config.representation,
            embeddings_trainable=encoder_config.embeddings_trainable,
            pretrained_embeddings=encoder_config.pretrained_embeddings,
            force_embedding_size=encoder_config.force_embedding_size,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )
        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            self.embed_weighted.output_shape[-1],
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

    @staticmethod
    def get_schema_cls():
        return BagEmbedWeightedConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([len(self.vocab)])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x vocab size], type torch.int32

        :param return: embeddings of shape [batch x embed size], type torch.float32
        """
        hidden = self.embed_weighted(inputs)
        hidden = self.fc_stack(hidden)

        return hidden
