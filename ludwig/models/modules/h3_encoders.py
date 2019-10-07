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

import tensorflow as tf

from ludwig.models.modules.embedding_modules import Embed
from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.recurrent_modules import RecurrentStack
from ludwig.models.modules.reduction_modules import reduce_sum, reduce_sequence

logger = logging.getLogger(__name__)


class H3Embed:

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=10,
            norm=None,
            activation='relu',
            dropout=False,
            initializer=None,
            regularize=True,
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
        self.embedding_size = embedding_size
        self.reduce_output = reduce_output

        self.embed_mode = Embed(
            [str(i) for i in range(3)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_edge = Embed(
            [str(i) for i in range(7)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_resolution = Embed(
            [str(i) for i in range(16)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_base_cell = Embed(
            [str(i) for i in range(122)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_cells = Embed(
            [str(i) for i in range(8)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=True,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_activation=activation,
            default_norm=norm,
            default_dropout=dropout,
            default_regularize=regularize,
            default_initializer=initializer
        )

    def __call__(
            self,
            input_vector,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        """
            :param input_vector: The input vector fed into the encoder.
                   Shape: [batch x 19], type tf.int8
            :type input_vector: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout_rate: Tensor (tf.float) of the probability of dropout
            :type dropout_rate: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        with tf.compat.v1.variable_scope('mode', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_mode, _ = self.embed_mode(
                input_vector[:, 0:1],
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('edge', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_edge, _ = self.embed_edge(
                input_vector[:, 1:2],
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('resolution', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_resolution, _ = self.embed_resolution(
                input_vector[:, 2:3],
                regularizer,
                dropout_rate,
                is_training=True
            )
        with tf.compat.v1.variable_scope('base_cell', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_base_cell, _ = self.embed_base_cell(
                input_vector[:, 3:4],
                regularizer,
                dropout_rate,
                is_training=True
            )
        with tf.compat.v1.variable_scope('cells', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_cells, _ = self.embed_cells(
                input_vector[:, 4:],
                regularizer,
                dropout_rate,
                is_training=is_training
            )

        # ================ Masking ================
        resolution = input_vector[:, 2]
        mask = tf.cast(
            tf.expand_dims(tf.sequence_mask(resolution, 15), -1),
            dtype=tf.float32
        )
        masked_embedded_cells = embedded_cells * mask

        # ================ Reduce ================
        concatenated = tf.concat(
            [embedded_mode, embedded_edge, embedded_resolution,
             embedded_base_cell, masked_embedded_cells],
            axis=1)

        hidden = reduce_sequence(concatenated, self.reduce_output)

        # ================ FC Stack ================
        hidden_size = hidden.shape.as_list()[-1]
        logger.debug('  flatten hidden: {0}'.format(hidden))

        hidden = self.fc_stack(
            hidden,
            hidden_size,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )
        hidden_size = hidden.shape.as_list()[-1]

        return hidden, hidden_size


class H3WeightedSum:

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            should_softmax=False,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=10,
            norm=None,
            activation='relu',
            dropout=False,
            initializer=None,
            regularize=True,
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
        self.should_softmax = should_softmax

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize,
            reduce_output=None,
        )

        self.weights = tf.compat.v1.get_variable(
            'weights',
            [19, 1],
            initializer=initializer
        )

        self.fc_stack = FCStack(
            layers=fc_layers,
            num_layers=num_fc_layers,
            default_fc_size=fc_size,
            default_activation=activation,
            default_norm=norm,
            default_dropout=dropout,
            default_regularize=regularize,
            default_initializer=initializer
        )

    def __call__(
            self,
            input_vector,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        """
            :param input_vector: The input vector fed into the encoder.
                   Shape: [batch x 19], type tf.int8
            :type input_vector: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout_rate: Tensor (tf.float) of the probability of dropout
            :type dropout_rate: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        embedded_h3, embedding_size = self.h3_embed(
            input_vector,
            regularizer,
            dropout_rate,
            is_training=is_training
        )

        # ================ Weighted Sum ================
        if self.should_softmax:
            weights = tf.nn.softmax(self.weights)
        else:
            weights = self.weights

        hidden = reduce_sum(embedded_h3 * weights)

        # ================ FC Stack ================
        hidden_size = hidden.shape.as_list()[-1]
        logger.debug('  flatten hidden: {0}'.format(hidden))

        hidden = self.fc_stack(
            hidden,
            hidden_size,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )
        hidden_size = hidden.shape.as_list()[-1]

        return hidden, hidden_size


class H3RNN:

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            num_layers=1,
            state_size=10,
            cell_type='rnn',
            bidirectional=False,
            dropout=False,
            initializer=None,
            regularize=True,
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
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
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
        self.embedding_size = embedding_size

        self.h3_embed = H3Embed(
            embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize,
            reduce_output=None
        )

        self.recurrent_stack = RecurrentStack(
            state_size=state_size,
            cell_type=cell_type,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            regularize=regularize,
            reduce_output=reduce_output
        )

    def __call__(
            self,
            input_vector,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        """
            :param input_vector: The input vector fed into the encoder.
                   Shape: [batch x 19], type tf.int8
            :type input_vector: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout_rate: Tensor (tf.float) of the probability of dropout
            :type dropout_rate: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        embedded_h3, _ = self.h3_embed(
            input_vector,
            regularizer,
            dropout_rate,
            is_training=is_training
        )

        # ================ RNN ================
        hidden, hidden_size = self.recurrent_stack(
            embedded_h3,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )

        return hidden, hidden_size
