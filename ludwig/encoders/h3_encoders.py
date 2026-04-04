#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, ENCODER_OUTPUT_STATE, H3, H3_VECTOR_LENGTH, MAX_H3_RESOLUTION
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.modules.embedding_modules import Embed, EmbedSequence
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.initializer_modules import get_initializer
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.h3_encoders import H3EmbedConfig, H3RNNConfig, H3WeightedSumConfig
from ludwig.utils import torch_utils

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("embed", H3)
class H3Embed(Encoder):
    """Encodes H3 geospatial indices using learned categorical embeddings.

    H3 is a hierarchical hexagonal geospatial indexing system (Uber, 2018). Each H3
    index is decomposed into components: mode, edge, resolution, base cell, and up to
    15 resolution cells. Each component is embedded via a learned lookup table, the
    resolution cells are masked to the actual resolution, and the sequence is reduced
    (default: sum) and passed through an optional FC stack.

    Use this encoder for geospatial features encoded as H3 indices. It captures the
    hierarchical structure of H3 at multiple resolutions.

    Reference: https://h3geo.org/
    """

    def __init__(
        self,
        embedding_size: int = 10,
        embeddings_on_cpu: bool = False,
        fc_layers: list | None = None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: str = None,
        norm_params: dict = None,
        activation: str = "relu",
        dropout: float = 0,
        reduce_output: str = "sum",
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.embedding_size = embedding_size
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

        logger.debug("  mode Embed")
        self.embed_mode = Embed(
            [str(i) for i in range(3)],
            embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        logger.debug("  edge Embed")
        self.embed_edge = Embed(
            [str(i) for i in range(7)],
            embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        logger.debug("  resolution Embed")
        self.embed_resolution = Embed(
            [str(i) for i in range(16)],
            embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        logger.debug("  base cell Embed")
        self.embed_base_cell = Embed(
            [str(i) for i in range(122)],
            embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        logger.debug("  cells Embed")
        self.embed_cells = EmbedSequence(
            [str(i) for i in range(8)],
            embedding_size,
            max_sequence_length=(H3_VECTOR_LENGTH - 4),
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=embedding_size,
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

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        """Encode an H3 feature vector.

        Args:
            inputs: Tensor of shape [batch, H3_VECTOR_LENGTH] with dtype int,
                containing [mode, edge, resolution, base_cell, cell_0, ..., cell_14].

        Returns:
            Dictionary with ENCODER_OUTPUT key mapping to tensor of shape [batch, output_size].
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
        mask = torch.unsqueeze(torch_utils.sequence_mask(resolution, MAX_H3_RESOLUTION), dim=-1).float()
        # Batch size X 15(max resolution) X embedding size
        masked_embedded_cells = embedded_cells * mask

        # ================ Reduce ================
        # Batch size X H3_VECTOR_LENGTH X embedding size
        concatenated = torch.cat(
            [embedded_mode, embedded_edge, embedded_resolution, embedded_base_cell, masked_embedded_cells], dim=1
        )

        hidden = self.reduce_sequence(concatenated)

        # ================ FC Stack ================
        hidden = self.fc_stack(hidden)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return H3EmbedConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_VECTOR_LENGTH])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@DeveloperAPI
@register_encoder("weighted_sum", H3)
class H3WeightedSum(Encoder):
    """Encodes H3 indices using a learned weighted sum over component embeddings.

    This encoder first embeds all H3 components using ``H3Embed`` (with no reduction),
    then computes a weighted sum across the component dimension using learned (or
    optionally softmax-normalized) weights. The result is passed through an FC stack.

    Compared to ``H3Embed`` with sum reduction, this encoder learns per-component
    importance weights, allowing the model to attend more to certain hierarchy levels
    (e.g., base cell vs. fine-grained resolution cells).
    """

    def __init__(
        self,
        embedding_size: int = 10,
        embeddings_on_cpu: bool = False,
        should_softmax: bool = False,
        fc_layers: list | None = None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: str | None = None,
        norm_params: dict = None,
        activation: str = "relu",
        dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.should_softmax = should_softmax
        self.sum_sequence_reducer = SequenceReducer(reduce_mode="sum")

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            reduce_output="None",
        )

        self.register_buffer(
            "aggregation_weights", torch.Tensor(get_initializer(weights_initializer)([H3_VECTOR_LENGTH, 1]))
        )

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=self.h3_embed.output_shape[0],
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

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        """Encode an H3 feature vector using a learned weighted sum.

        Args:
            inputs: Tensor of shape [batch, H3_VECTOR_LENGTH] with dtype int.

        Returns:
            Dictionary with ENCODER_OUTPUT key mapping to tensor of shape [batch, output_size].
        """
        # ================ Embeddings ================
        input_vector = inputs
        embedded_h3 = self.h3_embed(input_vector)

        # ================ Weighted Sum ================
        if self.should_softmax:
            weights = torch.softmax(self.aggregation_weights, dim=None)
        else:
            weights = self.aggregation_weights

        hidden = self.sum_sequence_reducer(embedded_h3[ENCODER_OUTPUT] * weights)

        # ================ FC Stack ================
        hidden = self.fc_stack(hidden)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return H3WeightedSumConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_VECTOR_LENGTH])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@DeveloperAPI
@register_encoder("rnn", H3)
class H3RNN(Encoder):
    """Encodes H3 indices by treating the component sequence as a time series for an RNN.

    This encoder first embeds all H3 components using ``H3Embed`` (with no reduction),
    then feeds the resulting sequence of embeddings through a recurrent neural network
    (RNN, LSTM, or GRU). This allows the model to capture sequential dependencies
    across the H3 hierarchy levels (mode -> edge -> resolution -> base cell -> cells).

    Use this encoder when the sequential/hierarchical structure of H3 indices is
    important for the task. For simpler pooling-based approaches, use ``H3Embed``
    or ``H3WeightedSum``.
    """

    def __init__(
        self,
        embedding_size: int = 10,
        embeddings_on_cpu: bool = False,
        num_layers: int = 1,
        hidden_size: int = 10,
        cell_type: str = "rnn",
        bidirectional: bool = False,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        unit_forget_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        reduce_output: str = "last",
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.embedding_size = embedding_size

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            reduce_output="None",
        )

        logger.debug("  RecurrentStack")
        self.recurrent_stack = RecurrentStack(
            input_size=self.h3_embed.output_shape[0],
            max_sequence_length=H3_VECTOR_LENGTH,
            hidden_size=hidden_size,
            cell_type=cell_type,
            num_layers=num_layers,
            bidirectional=bidirectional,
            use_bias=use_bias,
            dropout=recurrent_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        """Encode an H3 feature vector through an RNN.

        Args:
            inputs: Tensor of shape [batch, H3_VECTOR_LENGTH] with dtype int.

        Returns:
            Dictionary with ENCODER_OUTPUT and ENCODER_OUTPUT_STATE keys.
        """
        # ================ Embeddings ================
        embedded_h3 = self.h3_embed(inputs)

        # ================ RNN ================
        hidden, final_state = self.recurrent_stack(embedded_h3[ENCODER_OUTPUT])

        return {ENCODER_OUTPUT: hidden, ENCODER_OUTPUT_STATE: final_state}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return H3RNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_VECTOR_LENGTH])

    @property
    def output_shape(self) -> torch.Size:
        return self.recurrent_stack.output_shape
