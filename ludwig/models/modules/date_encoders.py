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

import math
import tensorflow as tf

from ludwig.models.modules.embedding_modules import Embed
from ludwig.models.modules.fully_connected_modules import FCStack

logger = logging.getLogger(__name__)


class DateEmbed:

    def __init__(
            self,
            embedding_size=10,
            embeddings_on_cpu=False,
            dropout=False,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=10,
            norm=None,
            activation='relu',
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
        self.year_fc = FCStack(
            num_layers=1,
            default_fc_size=1,
            default_activation=None,
            default_norm=None,
            default_dropout=dropout,
            default_regularize=regularize,
            default_initializer=initializer
        )
        self.embed_month = Embed(
            [str(i) for i in range(12)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_day = Embed(
            [str(i) for i in range(31)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_weekday = Embed(
            [str(i) for i in range(7)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_yearday = Embed(
            [str(i) for i in range(366)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_hour = Embed(
            [str(i) for i in range(24)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_minute = Embed(
            [str(i) for i in range(60)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )
        self.embed_second = Embed(
            [str(i) for i in range(60)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
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
        with tf.compat.v1.variable_scope('year', reuse=tf.compat.v1.AUTO_REUSE):
            scaled_year = self.year_fc(
                tf.cast(input_vector[:, 0:1], tf.float32),
                1,
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('month', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_month, _ = self.embed_month(
                input_vector[:, 1] - 1,
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('day', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_day, _ = self.embed_day(
                input_vector[:, 2] - 1,
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('weekday', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_weekday, _ = self.embed_weekday(
                input_vector[:, 3],
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('yearday', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_yearday, _ = self.embed_yearday(
                input_vector[:, 4] - 1,
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('hour', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_hour, _ = self.embed_hour(
                input_vector[:, 5],
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('minute', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_minute, _ = self.embed_minute(
                input_vector[:, 6],
                regularizer,
                dropout_rate,
                is_training=is_training
            )
        with tf.compat.v1.variable_scope('second', reuse=tf.compat.v1.AUTO_REUSE):
            embedded_second, _ = self.embed_second(
                input_vector[:, 7],
                regularizer,
                dropout_rate,
                is_training=is_training
            )

        periodic_second_of_day = tf.sin(
            tf.cast(input_vector[:, 8:9], dtype=tf.float32)
            * (2 * math.pi / 86400)
        )

        hidden = tf.concat(
            [scaled_year, embedded_month, embedded_day,
             embedded_weekday, embedded_yearday,
             embedded_hour, embedded_minute, embedded_second,
             periodic_second_of_day],
            axis=1
        )

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


class DateWave:

    def __init__(
            self,
            dropout=False,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=8,
            norm=None,
            activation='relu',
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
        self.year_fc = FCStack(
            num_layers=1,
            default_fc_size=1,
            default_activation=None,
            default_norm=None,
            default_dropout=dropout,
            default_regularize=regularize,
            default_initializer=initializer
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
        input_vector = tf.cast(input_vector, tf.float32)
        scaled_year = self.year_fc(
            input_vector[:, 0:1],
            1,
            regularizer,
            dropout_rate,
            is_training=is_training
        )
        periodic_month = tf.sin(input_vector[:, 1:2] * (2 * math.pi / 12))
        periodic_day = tf.sin(input_vector[:, 2:3] * (2 * math.pi / 31))
        periodic_weekday = tf.sin(input_vector[:, 3:4] * (2 * math.pi / 7))
        periodic_yearday = tf.sin(input_vector[:, 4:5] * (2 * math.pi / 366))
        periodic_hour = tf.sin(input_vector[:, 5:6] * (2 * math.pi / 24))
        periodic_minute = tf.sin(input_vector[:, 6:7] * (2 * math.pi / 60))
        periodic_second = tf.sin(input_vector[:, 7:8] * (2 * math.pi / 60))
        periodic_second_of_day = tf.sin(
            input_vector[:, 8:9] * (2 * math.pi / 86400)
        )

        hidden = tf.concat(
            [scaled_year, periodic_month, periodic_day,
             periodic_weekday, periodic_yearday,
             periodic_hour, periodic_minute, periodic_second,
             periodic_second_of_day],
            axis=1)

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
