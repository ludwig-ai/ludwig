#! /usr/bin/env python
# coding=utf-8
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
from abc import ABC
from typing import Dict

import torch

from ludwig.encoders.base import Encoder
from ludwig.utils import torch_utils
from ludwig.utils.registry import Registry, register
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.initializer_modules import get_initializer
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer

logger = logging.getLogger(__name__)

ENCODER_REGISTRY = Registry()

# TODO: Share this with h3_feature.H3_VECTOR_LENGTH
H3_INPUT_SIZE = 19


class H3Encoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register(name='embed')
class H3Embed(H3Encoder):

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=10,
            use_bias=True,
            weights_initializer='xavier_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            norm=None,
            norm_params=None,
            activation='relu',
            dropout=0,
            reduce_output='sum',
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
        logger.debug(' {}'.format(self.name))

        self.embedding_size = embedding_size
        self.reduce_output = reduce_output
        self.sum_sequence_reducer = SequenceReducer(reduce_mode=reduce_output)

        logger.debug('  mode Embed')
        self.embed_mode = Embed(
            [str(i) for i in range(3)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  edge Embed')
        self.embed_edge = Embed(
            [str(i) for i in range(7)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  resolution Embed')
        self.embed_resolution = Embed(
            [str(i) for i in range(16)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  base cell Embed')
        self.embed_base_cell = Embed(
            [str(i) for i in range(122)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  cells Embed')
        self.embed_cells = Embed(
            [str(i) for i in range(8)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  FCStack')
        self.fc_size = fc_size
        self.fc_stack = FCStack(
            first_layer_input_size=H3_INPUT_SIZE,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
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
        input_vector = inputs.type(torch.IntTensor)

        # ================ Embeddings ================
        embedded_mode = self.embed_mode(
            input_vector[:, 0:1],
        )
        embedded_edge = self.embed_edge(
            input_vector[:, 1:2],
        )
        embedded_resolution = self.embed_resolution(
            input_vector[:, 2:3],
        )
        embedded_base_cell = self.embed_base_cell(
            input_vector[:, 3:4],
        )
        embedded_cells = self.embed_cells(
            input_vector[:, 4:].unsqueeze(1),
        )

        # ================ Masking ================
        resolution = input_vector[:, 2]
        mask = torch.unsqueeze(
            torch_utils.sequence_mask(resolution, 15), -1).type(
            torch.FloatTensor)
        masked_embedded_cells = embedded_cells * mask
        logger.error(
            f'resolution.size(): {resolution.size()}')
        logger.error(
            f'mask.size(): {mask.size()}')
        logger.error(
            f'embedded_cells.size(): {embedded_cells.size()}')
        logger.error(
            f'masked_embedded_cells.size(): {masked_embedded_cells.size()}')

        # ================ Reduce ================
        concatenated = torch.cat(
            [embedded_mode, embedded_edge, embedded_resolution,
             embedded_base_cell, masked_embedded_cells],
            dim=1)

        hidden = self.sum_sequence_reducer(concatenated)

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))
        hidden = self.fc_stack(hidden)

        return {'encoder_output': hidden}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@register(name='weighted_sum')
class H3WeightedSum(H3Encoder):

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            should_softmax=False,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=10,
            use_bias=True,
            weights_initializer='xavier_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            norm=None,
            norm_params=None,
            activation='relu',
            dropout=0,
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
        logger.debug(' {}'.format(self.name))

        self.should_softmax = should_softmax
        self.sum_sequence_reducer = SequenceReducer(reduce_mode='sum')

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            reduce_output='None'
        )

        self.aggregation_weights = torch.Tensor(
            get_initializer(weights_initializer)([H3_INPUT_SIZE, 1])
        )

        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            first_layer_input_size=self.h3_embed.fc_size,
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
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

        hidden = self.sum_sequence_reducer(
            embedded_h3['encoder_output'] * weights)

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))
        hidden = self.fc_stack(hidden)

        return {'encoder_output': hidden}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@register(name='rnn')
class H3RNN(H3Encoder):

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            num_layers=1,
            state_size=10,
            cell_type='rnn',
            bidirectional=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            unit_forget_bias=True,
            weights_initializer='xavier_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            weights_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            reduce_output='last'
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
                   refer to TensorFlow's documentation. We suggest to use the
                   `block` variants on CPU and the `cudnn` variants on GPU
                   because of their increased speed.
            :type cell_type: str
            :param state_size: the size of the state of the rnn.
            :type state_size: Integer
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
            :param weights_regularizer: regularizer applied to the weights
                   (kernel) matrix
            :type weights_regularizer: string
            :param recurrent_regularizer: Regularizer for the recurrent weights
                   matrix
            :type recurrent_regularizer: string
            :param bias_regularizer: regularized function applied to bias
                vector.
            :type bias_regularizer: string
            :param activity_regularizer: Regularizer applied to the output of
                the layer (activation)
            :type activity_regularizer: string
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: float
            :param recurrent_dropout: Float between 0.0 and 1.0.  Fraction of
                   the units to drop for the linear transformation of the
                   recurrent state.
            :type recurrent_dropout: float
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimension if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super().__init__()
        logger.debug(' {}'.format(self.name))

        self.embedding_size = embedding_size

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            reduce_output='None'
        )

        logger.debug('  RecurrentStack')
        self.recurrent_stack = RecurrentStack(
            state_size=state_size,
            cell_type=cell_type,
            num_layers=num_layers,
            bidirectional=bidirectional,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
            weights_initializer=weights_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            reduce_output=reduce_output
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
        hidden, final_state = self.recurrent_stack(
            embedded_h3['encoder_output']
        )

        return {
            'encoder_output': hidden,
            'encoder_output_state': final_state
        }

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([H3_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape
