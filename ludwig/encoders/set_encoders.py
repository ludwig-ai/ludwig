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
from typing import Any, Dict, List, Optional

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SET
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import EmbedSet
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.set_encoders import SetSparseEncoderConfig

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("embed", SET)
class SetSparseEncoder(Encoder):
    def __init__(
        self,
        vocab: List[str],
        representation: str = "dense",
        embedding_size: int = 50,
        embeddings_trainable: bool = True,
        pretrained_embeddings: Optional[str] = None,
        embeddings_on_cpu: bool = False,
        fc_layers=None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: Optional[str] = None,
        norm_params: Optional[Dict[str, Any]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.vocab_size = len(vocab)

        logger.debug("  Embed")
        self.embed = EmbedSet(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        logger.debug("  FCStack")
        # TODO(shreya): Make sure this is updated when FCStack is updated
        self.fc_stack = FCStack(
            first_layer_input_size=self.embed.output_shape[-1],
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_output_size=output_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
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
