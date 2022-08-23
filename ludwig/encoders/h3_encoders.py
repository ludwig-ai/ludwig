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
from typing import Dict

import torch

from ludwig.constants import H3
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import Embed, EmbedSequence
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.initializer_modules import get_initializer
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.encoders.h3_encoders import H3EmbedConfig, H3RNNConfig, H3WeightedSumConfig
from ludwig.utils import torch_utils

logger = logging.getLogger(__name__)

# TODO: Share this with h3_feature.H3_VECTOR_LENGTH
H3_INPUT_SIZE = 19


@register_encoder("embed", H3)
class H3Embed(Encoder):
    def __init__(self, encoder_config: H3EmbedConfig = H3EmbedConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        self.embedding_size = encoder_config.embedding_size
        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=self.reduce_output)

        logger.debug("  mode Embed")
        self.embed_mode = Embed(
            [str(i) for i in range(3)],
            self.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  edge Embed")
        self.embed_edge = Embed(
            [str(i) for i in range(7)],
            self.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  resolution Embed")
        self.embed_resolution = Embed(
            [str(i) for i in range(16)],
            self.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  base cell Embed")
        self.embed_base_cell = Embed(
            [str(i) for i in range(122)],
            self.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  cells Embed")
        self.embed_cells = EmbedSequence(
            [str(i) for i in range(8)],
            self.embedding_size,
            max_sequence_length=(H3_INPUT_SIZE - 4),
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=self.embedding_size,
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

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The input vector fed into the encoder.
               Shape: [batch x H3_INPUT_SIZE], type torch.int8
        :type inputs: Tensor
        """
        input_vector = inputs.int()

        # ================ Embeddings ================
        embedded_mode = self.embed_mode(input_vector[:, 0:1]).unsqueeze(1)
        embedded_edge = self.embed_edge(input_vector[:, 1:2]).unsqueeze(1)
        embedded_resolution = self.embed_resolution(input_vector[:, 2:3]).unsqueeze(1)
        embedded_base_cell = self.embed_base_cell(input_vector[:, 3:4]).unsqueeze(1)
        embedded_cells = self.embed_cells(input_vector[:, 4:])

        # ================ Masking ================
        # Mask out cells beyond the resolution of interest.
        resolution = input_vector[:, 2]
        mask = torch.unsqueeze(torch_utils.sequence_mask(resolution, 15), dim=-1).float()
        # Batch size X 15(max resolution) X embedding size
        masked_embedded_cells = embedded_cells * mask

        # ================ Reduce ================
        # Batch size X H3_INPUT_SIZE X embedding size
        concatenated = torch.cat(
            [embedded_mode, embedded_edge, embedded_resolution, embedded_base_cell, masked_embedded_cells], dim=1
        )

        hidden = self.reduce_sequence(concatenated)

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))
        hidden = self.fc_stack(hidden)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return H3EmbedConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@register_encoder("weighted_sum", H3)
class H3WeightedSum(Encoder):
    def __init__(self, encoder_config: H3WeightedSumConfig = H3WeightedSumConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")

        self.should_softmax = encoder_config.should_softmax
        self.sum_sequence_reducer = SequenceReducer(reduce_mode="sum")

        self.h3_embed = H3Embed(
            encoder_config.embedding_size,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            weights_initializer=encoder_config.weights_initializer,
            bias_initializer=encoder_config.bias_initializer,
            reduce_output="None",
        )

        self.register_buffer(
            "aggregation_weights", torch.Tensor(get_initializer(encoder_config.weights_initializer)([H3_INPUT_SIZE, 1]))
        )

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=self.h3_embed.output_shape[0],
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

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The input vector fed into the encoder.
               Shape: [batch x H3_INPUT_SIZE], type torch.int8
        :type inputs: Tensor
        """
        # ================ Embeddings ================
        input_vector = inputs
        embedded_h3 = self.h3_embed(input_vector)

        # ================ Weighted Sum ================
        if self.should_softmax:
            weights = torch.softmax(self.aggregation_weights, dim=None)
        else:
            weights = self.aggregation_weights

        hidden = self.sum_sequence_reducer(embedded_h3["encoder_output"] * weights)

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))
        hidden = self.fc_stack(hidden)

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return H3WeightedSumConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@register_encoder("rnn", H3)
class H3RNN(Encoder):
    def __init__(self, encoder_config: H3RNNConfig = H3RNNConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")

        self.embedding_size = encoder_config.embedding_size

        self.h3_embed = H3Embed(
            self.embedding_size,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            weights_initializer=encoder_config.weights_initializer,
            bias_initializer=encoder_config.bias_initializer,
            reduce_output="None",
        )

        logger.debug("  RecurrentStack")
        self.recurrent_stack = RecurrentStack(
            input_size=self.h3_embed.output_shape[0],
            max_sequence_length=H3_INPUT_SIZE,
            hidden_size=encoder_config.hidden_size,
            cell_type=encoder_config.cell_type,
            num_layers=encoder_config.num_layers,
            bidirectional=encoder_config.bidirectional,
            use_bias=encoder_config.use_bias,
            dropout=encoder_config.recurrent_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The input vector fed into the encoder.
               Shape: [batch x H3_INPUT_SIZE], type torch.int8
        :type inputs: Tensor
        """

        # ================ Embeddings ================
        embedded_h3 = self.h3_embed(inputs)

        # ================ RNN ================
        hidden, final_state = self.recurrent_stack(embedded_h3["encoder_output"])

        return {"encoder_output": hidden, "encoder_output_state": final_state}

    @staticmethod
    def get_schema_cls():
        return H3RNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.recurrent_stack.output_shape
