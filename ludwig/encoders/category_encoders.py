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
from typing import Optional, Union

import torch
from torch import nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import Embed
from ludwig.schema.encoders.category_encoders import (
    CategoricalEmbedConfig,
    CategoricalOneHotEncoderConfig,
    CategoricalPassthroughEncoderConfig,
    CategoricalSparseConfig,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("passthrough", [CATEGORY])
class CategoricalPassthroughEncoder(Encoder):
    def __init__(self, input_size=1, encoder_config=None, **kwargs):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")
        self.input_size = input_size

    def forward(self, inputs, mask=None):
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch x 1]
        """
        return inputs.float()

    @staticmethod
    def get_schema_cls():
        return CategoricalPassthroughEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape

    def get_embedding_layer(self) -> nn.Module:
        return self


@DeveloperAPI
@register_encoder("dense", CATEGORY)
class CategoricalEmbedEncoder(Encoder):
    def __init__(
        self,
        vocab: list[str],
        embedding_size: int = 50,
        embeddings_trainable: bool = True,
        pretrained_embeddings: Optional[str] = None,
        embeddings_on_cpu: bool = False,
        dropout: float = 0.0,
        embedding_initializer: Optional[Union[str, dict]] = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Embed")
        self.embed = Embed(
            vocab=vocab,
            embedding_size=embedding_size,
            representation="dense",
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=embedding_initializer,
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


@DeveloperAPI
@register_encoder("sparse", CATEGORY)
class CategoricalSparseEncoder(Encoder):
    def __init__(
        self,
        vocab: list[str],
        embeddings_trainable: bool = False,
        pretrained_embeddings: Optional[str] = None,
        embeddings_on_cpu: bool = False,
        dropout: float = 0.0,
        embedding_initializer: Optional[Union[str, dict]] = None,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Embed")
        self.embed = Embed(
            vocab=vocab,
            embedding_size=len(vocab),
            representation="sparse",
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=embedding_initializer,
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


@DeveloperAPI
@register_encoder("onehot", [CATEGORY])
class CategoricalOneHotEncoder(Encoder):
    def __init__(
        self,
        vocab: list[str],
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")
        self.vocab_size = len(vocab)

    def forward(self, inputs, mask=None):
        """
        :param inputs: The inputs fed into the encoder.
               Shape: [batch, 1] or [batch]
        """
        t = inputs.reshape(-1).long()
        # the output of this must be a float so that it can be concatenated with other
        # encoder outputs and passed to dense layers in the combiner, decoder, etc.
        return torch.nn.functional.one_hot(t, num_classes=self.vocab_size).float()

    @staticmethod
    def get_schema_cls():
        return CategoricalOneHotEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.vocab_size])

    def get_embedding_layer(self) -> nn.Module:
        return self
