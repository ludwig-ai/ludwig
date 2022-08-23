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

from ludwig.constants import CATEGORY
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import Embed
from ludwig.schema.encoders.category_encoders import CategoricalEmbedConfig, CategoricalSparseConfig

logger = logging.getLogger(__name__)


@register_encoder("dense", CATEGORY)
class CategoricalEmbedEncoder(Encoder):
    def __init__(self, encoder_config: CategoricalEmbedConfig = CategoricalEmbedConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")
        logger.debug("  Embed")
        self.embed = Embed(
            vocab=encoder_config.vocab,
            embedding_size=encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=encoder_config.embeddings_trainable,
            pretrained_embeddings=encoder_config.pretrained_embeddings,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.embedding_initializer,
        )
        self.embedding_size = self.embed.embedding_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type torch.int32

        :param return: embeddings of shape [batch x embed size], type torch.float32
        """
        embedded = self.embed(inputs)
        return embedded

    @staticmethod
    def get_schema_cls():
        return CategoricalEmbedConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.embedding_size])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])


@register_encoder("sparse", CATEGORY)
class CategoricalSparseEncoder(Encoder):
    def __init__(self, encoder_config: CategoricalSparseConfig):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")
        logger.debug("  Embed")
        self.embed = Embed(
            vocab=encoder_config.vocab,
            embedding_size=len(encoder_config.vocab),
            representation="sparse",
            embeddings_trainable=encoder_config.embeddings_trainable,
            pretrained_embeddings=encoder_config.pretrained_embeddings,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.embedding_initializer,
        )
        self.embedding_size = self.embed.embedding_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1], type torch.int32

        :param return: embeddings of shape [batch x embed size], type torch.float32
        """
        embedded = self.embed(inputs)
        return embedded

    @staticmethod
    def get_schema_cls():
        return CategoricalSparseConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.embedding_size])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])
