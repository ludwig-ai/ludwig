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
from typing import Dict, List, Optional

import torch

from ludwig.api_annotations import DeveloperAPI
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


@DeveloperAPI
@register_encoder("embed", H3)
class H3Embed(Encoder):
    def __init__(
        self,
        embedding_size: int = 10,
        embeddings_on_cpu: bool = False,
        fc_layers: Optional[List] = None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: str = None,
        norm_params: Dict = None,
        activation: str = "relu",
        dropout: float = 0,
        reduce_output: str = "sum",
        encoder_config=None,
        **kwargs,
    ):
        """
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
               for the `sparse` encoding, where `vocabulary_size` is
               the number of different strings appearing in the training set
               in the column the feature is named after (plus 1 for
               `<UNK>`).
        :type embedding_size: Integer
        :param embeddings_on_cpu: by default embeddings matrices are stored
               on GPU memory if a GPU is used, as it allows
               for faster access, but in some cases the embedding matrix
               may be really big and this parameter forces the placement
               of the embedding matrix in regular memory and the CPU is used
               to resolve them, slightly slowing down the process
               as a result of data transfer between CPU and GPU memory.
        :param dropout: determines if there should be a dropout layer before
               returning the encoder output.
        :type dropout: Boolean
        """
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
            max_sequence_length=(H3_INPUT_SIZE - 4),
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


@DeveloperAPI
@register_encoder("weighted_sum", H3)
class H3WeightedSum(Encoder):
    def __init__(
        self,
        embedding_size: int = 10,
        embeddings_on_cpu: bool = False,
        should_softmax: bool = False,
        fc_layers: Optional[List] = None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: Optional[str] = None,
        norm_params: Dict = None,
        activation: str = "relu",
        dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        """
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
               for the `sparse` encoding, where `vocabulary_size` is
               the number of different strings appearing in the training set
               in the column the feature is named after (plus 1 for
               `<UNK>`).
        :type embedding_size: Integer
        :param embeddings_on_cpu: by default embeddings matrices are stored
               on GPU memory if a GPU is used, as it allows
               for faster access, but in some cases the embedding matrix
               may be really big and this parameter forces the placement
               of the embedding matrix in regular memory and the CPU is used
               to resolve them, slightly slowing down the process
               as a result of data transfer between CPU and GPU memory.
        :param dropout: determines if there should be a dropout layer before
               returning the encoder output.
        :type dropout: Boolean
        """
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
            "aggregation_weights", torch.Tensor(get_initializer(weights_initializer)([H3_INPUT_SIZE, 1]))
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


@DeveloperAPI
@register_encoder("rnn", H3)
class H3RNN(Encoder):
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
        """
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
               for the `sparse` encoding, where `vocabulary_size` is
               the number of different strings appearing in the training set
               in the column the feature is named after (plus 1 for
               `<UNK>`).
        :type embedding_size: Integer
        :param embeddings_on_cpu: by default embeddings matrices are stored
               on GPU memory if a GPU is used, as it allows
               for faster access, but in some cases the embedding matrix
               may be really big and this parameter forces the placement
               of the embedding matrix in regular memory and the CPU is used
               to resolve them, slightly slowing down the process
               as a result of data transfer between CPU and GPU memory.
        :param num_layers: the number of stacked recurrent layers.
        :type num_layers: Integer
        :param cell_type: the type of recurrent cell to use.
               Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`,
               `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`.
               For reference about the differences between the cells please
               refer to PyTorch's documentation. We suggest to use the
               `block` variants on CPU and the `cudnn` variants on GPU
               because of their increased speed.
        :type cell_type: str
        :param hidden_size: the size of the state of the rnn.
        :type hidden_size: Integer
        :param bidirectional: if `True` two recurrent networks will perform
               encoding in the forward and backward direction and
               their outputs will be concatenated.
        :type bidirectional: Boolean
        :param activation: Activation function to use.
        :type activation: string
        :param recurrent_activation: Activation function to use for the
                recurrent step.
        :type recurrent_activation: string
        :param use_bias: bool determines where to use a bias vector
        :type use_bias: bool
        :param unit_forget_bias: if True add 1 to the bias forget gate at
               initialization.
        :type unit_forget_bias: bool
        :param weights_initializer: Initializer for the weights (aka kernel)
               matrix
        :type weights_initializer: string
        :param recurrent_initializer: Initializer for the recurrent weights
               matrix
        :type recurrent_initializer: string
        :param bias_initializer: Initializer for the bias vector
        :type bias_initializer: string
        :param dropout: determines if there should be a dropout layer before
               returning the encoder output.
        :type dropout: float
        :param recurrent_dropout: Dropout rate for the RNN encoder of the H3 embeddings.
        :type recurrent_dropout: float
        """
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
            max_sequence_length=H3_INPUT_SIZE,
            hidden_size=hidden_size,
            cell_type=cell_type,
            num_layers=num_layers,
            bidirectional=bidirectional,
            use_bias=use_bias,
            dropout=recurrent_dropout,
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
