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

import tensorflow as tf

from ludwig.encoders.base import Encoder
from ludwig.utils.registry import Registry, register
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.initializer_modules import get_initializer
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry()


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
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm=None,
            norm_params=None,
            activation='relu',
            dropout=0,
            reduce_output='sum',
            **kwargs
    ):
        """
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None`, the default
                   initialized of each variable is used (`glorot_uniform`
                   in most cases). Options are: `constant`, `identity`, `zeros`,
                    `ones`, `orthogonal`, `normal`, `uniform`,
                    `truncated_normal`, `variance_scaling`, `glorot_normal`,
                    `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                    `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                    Alternatively it is possible to specify a dictionary with
                    a key `type` that identifies the type of initialzier and
                    other keys for its parameters, e.g.
                    `{type: normal, mean: 0, stddev: 0}`.
                    To know the parameters of each initializer, please refer to
                    TensorFlow's documentation.
            :type initializer: str
            :param regularize: if `True` the embedding wieghts are added to
                   the set of weights that get reularized by a regularization
                   loss (if the `regularization_lambda` in `training`
                   is greater than 0).
            :type regularize: Boolean
        """
        super(H3Embed, self).__init__()
        logger.debug(' {}'.format(self.name))

        self.embedding_size = embedding_size
        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

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
        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
        )

    def call(
            self,
            inputs,
            training=None,
            mask=None
    ):
        """
            :param input_vector: The input vector fed into the encoder.
                   Shape: [batch x 19], type tf.int8
            :type input_vector: Tensor
            :param training: bool specifying if in training mode (important for dropout)
            :type training: bool
            :param mask: bool tensor encoding masked timesteps in the input
            :type mask: bool
         """
        input_vector = tf.cast(inputs, tf.int32)

        # ================ Embeddings ================
        embedded_mode = self.embed_mode(
            input_vector[:, 0:1],
            training=training,
            mask=mask
        )
        embedded_edge = self.embed_edge(
            input_vector[:, 1:2],
            training=training,
            mask=mask
        )
        embedded_resolution = self.embed_resolution(
            input_vector[:, 2:3],
            training=training,
            mask=mask
        )
        embedded_base_cell = self.embed_base_cell(
            input_vector[:, 3:4],
            training=training,
            mask=mask
        )
        embedded_cells = self.embed_cells(
            input_vector[:, 4:],
            training=training,
            mask=mask
        )

        # ================ Masking ================
        resolution = input_vector[:, 2]
        mask = tf.cast(
            tf.expand_dims(tf.sequence_mask(resolution, 15),
                           -1),
            dtype=tf.float32
        )
        masked_embedded_cells = embedded_cells * mask

        # ================ Reduce ================
        concatenated = tf.concat(
            [embedded_mode, embedded_edge, embedded_resolution,
             embedded_base_cell, masked_embedded_cells],
            axis=1)

        hidden = self.reduce_sequence(concatenated)

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))
        hidden = self.fc_stack(
            hidden,
            training=training,
            mask=mask
        )

        return {'encoder_output': hidden}


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
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # weights_constraint=None,
            # bias_constraint=None,
            norm=None,
            norm_params=None,
            activation='relu',
            dropout=0,
            **kwargs
    ):
        """
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None`, the default
                   initialized of each variable is used (`glorot_uniform`
                   in most cases). Options are: `constant`, `identity`, `zeros`,
                    `ones`, `orthogonal`, `normal`, `uniform`,
                    `truncated_normal`, `variance_scaling`, `glorot_normal`,
                    `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                    `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                    Alternatively it is possible to specify a dictionary with
                    a key `type` that identifies the type of initialzier and
                    other keys for its parameters, e.g.
                    `{type: normal, mean: 0, stddev: 0}`.
                    To know the parameters of each initializer, please refer to
                    TensorFlow's documentation.
            :type initializer: str
            :param regularize: if `True` the embedding wieghts are added to
                   the set of weights that get reularized by a regularization
                   loss (if the `regularization_lambda` in `training`
                   is greater than 0).
            :type regularize: Boolean
        """
        super(H3WeightedSum, self).__init__()
        logger.debug(' {}'.format(self.name))

        self.should_softmax = should_softmax
        self.reduce_sequence = SequenceReducer(reduce_mode='sum')

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            # weights_constraint=weights_constraint,
            # bias_constraint=bias_constraint,
            reduce_output=None
        )

        self.aggregation_weights = tf.Variable(
            get_initializer(weights_initializer)([19, 1])
        )

        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
        )

    def call(
            self,
            inputs,
            training=None,
            mask=None
    ):
        """
            :param input_vector: The input vector fed into the encoder.
                   Shape: [batch x 19], type tf.int8
            :type input_vector: Tensor
            :param training: bool specifying if in training mode (important for dropout)
            :type training: bool
            :param mask: bool tensor encoding masked timesteps in the input
            :type mask: bool
         """
        # ================ Embeddings ================
        input_vector = inputs
        embedded_h3 = self.h3_embed(
            input_vector,
            training=training,
            mask=mask
        )

        # ================ Weighted Sum ================
        if self.should_softmax:
            weights = tf.nn.softmax(self.aggregation_weights)
        else:
            weights = self.aggregation_weights

        hidden = self.reduce_sequence(embedded_h3['encoder_output'] * weights)

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))
        hidden = self.fc_stack(
            hidden,
            training=training,
            mask=mask
        )

        return {'encoder_output': hidden}


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
            weights_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            weights_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            reduce_output='last',
            **kwargs
    ):
        """
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param num_layers: the number of stacked recurrent layers.
            :type num_layers: Integer
            :param cell_type: the type of recurrent cell to use.
                   Avalable values are: `rnn`, `lstm`, `lstm_block`, `lstm`,
                   `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`.
                   For reference about the differences between the cells please
                   refer to TensorFlow's documentstion. We suggest to use the
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
                   (kernal) matrix
            :type weights_regularizer: string
            :param recurrent_regularizer: Regularizer for the recurrent weights
                   matrix
            :type recurrent_regularizer: string
            :param bias_regularizer: reguralizer function applied to biase vector.
            :type bias_regularizer: string
            :param activity_regularizer: Regularizer applied to the output of the
                   layer (activation)
            :type activity_regularizer: string
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: float
            :param recurrent_dropout: Float between 0.0 and 1.0.  Fraction of
                   the units to drop for the linear transformation of the
                   recurrent state.
            :type recurrent_dropout: float
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(H3RNN, self).__init__()
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
            # weights_constraint=weights_constraint,
            # bias_constraint=bias_constraint,
            reduce_output=None
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

    def call(
            self,
            inputs,
            training=None,
            mask=None
    ):
        """
            :param input_vector: The input vector fed into the encoder.
                   Shape: [batch x 19], type tf.int8
            :type input_vector: Tensor
            :param training: bool specifying if in training mode (important for dropout)
            :type training: bool
            :param mask: bool tensor encoding masked timesteps in the input
            :type mask: bool
         """

        # ================ Embeddings ================
        embedded_h3 = self.h3_embed(
            inputs,
            training=training,
            mask=mask
        )

        # ================ RNN ================
        hidden, final_state = self.recurrent_stack(
            embedded_h3['encoder_output'],
            training=training,
            mask=mask
        )

        return {
            'encoder_output': hidden,
            'encoder_output_state': final_state
        }
