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
from abc import ABC

import tensorflow as tf

from ludwig.encoders.base import Encoder
from ludwig.utils.registry import Registry, register
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry()


class DateEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register(name='embed')
class DateEmbed(DateEncoder):

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
            :param fc_layers: list of dictionaries containing the parameters of
                    all the fully connected layers
            :type fc_layers: List
            :param num_fc_layers: Number of stacked fully connected layers
            :type num_fc_layers: Integer
            :param fc_size: Size of each layer
            :type fc_size: Integer
            :param use_bias: bool determines where to use a bias vector
            :type use_bias: bool
            :param weights_initializer: Initializer for the weights (aka kernel)
                   matrix
            :type weights_initializer: string
            :param bias_initializer: Initializer for the bias vector
            :type bias_initializer: string
            :param weights_regularizer: regularizer applied to the weights
                   (kernal) matrix
            :type weights_regularizer: string
            :param bias_regularizer: reguralizer function applied to biase vector.
            :type bias_regularizer: string
            :param activity_regularizer: Regularizer applied to the output of the
                   layer (activation)
            :type activity_regularizer: string
            :param norm: type of normalization to use 'batch' or 'layer'
            :type norm: string, default None
            :param norm_params: parameters to pass to normalization function
            :type norm_params: dictionary
            :param activation: Activation function to use.
            :type activation: string
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: float

        """
        super(DateEmbed, self).__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  year FCStack')
        self.year_fc = FCStack(
            num_layers=1,
            default_fc_size=1,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
            default_norm=None,
            default_norm_params=None,
            default_activation=None,
            default_dropout=dropout,
        )

        logger.debug('  month Embed')
        self.embed_month = Embed(
            [str(i) for i in range(12)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  day Embed')
        self.embed_day = Embed(
            [str(i) for i in range(31)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  weekday Embed')
        self.embed_weekday = Embed(
            [str(i) for i in range(7)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  yearday Embed')
        self.embed_yearday = Embed(
            [str(i) for i in range(366)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  hour Embed')
        self.embed_hour = Embed(
            [str(i) for i in range(24)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  minute Embed')
        self.embed_minute = Embed(
            [str(i) for i in range(60)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

        logger.debug('  second Embed')
        self.embed_second = Embed(
            [str(i) for i in range(60)],
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
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
            :param mask: bool specifying masked values
            :type mask: bool
         """
        # ================ Embeddings ================
        input_vector = tf.cast(inputs, tf.int32)

        scaled_year = self.year_fc(
            tf.cast(input_vector[:, 0:1], tf.float32),
            training=training,
            mask=mask
        )
        embedded_month = self.embed_month(
            input_vector[:, 1] - 1,
            training=training,
            mask=mask
        )
        embedded_day = self.embed_day(
            input_vector[:, 2] - 1,
            training=training,
            mask=mask
        )
        embedded_weekday = self.embed_weekday(
            input_vector[:, 3],
            training=training,
            mask=mask
        )
        embedded_yearday = self.embed_yearday(
            input_vector[:, 4] - 1,
            training=training,
            mask=mask
        )
        embedded_hour = self.embed_hour(
            input_vector[:, 5],
            training=training,
            mask=mask
        )
        embedded_minute = self.embed_minute(
            input_vector[:, 6],
            training=training,
            mask=mask
        )
        embedded_second = self.embed_second(
            input_vector[:, 7],
            training=training,
            mask=mask
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
        # logger.debug('  flatten hidden: {0}'.format(hidden))

        hidden = self.fc_stack(
            hidden,
            training=training,
            mask=mask
        )

        return {'encoder_output': hidden}


@register(name='wave')
class DateWave(DateEncoder):

    def __init__(
            self,
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
            :param fc_layers: list of dictionaries containing the parameters of
                    all the fully connected layers
            :type fc_layers: List
            :param num_fc_layers: Number of stacked fully connected layers
            :type num_fc_layers: Integer
            :param fc_size: Size of each layer
            :type fc_size: Integer
            :param use_bias: bool determines where to use a bias vector
            :type use_bias: bool
            :param weights_initializer: Initializer for the weights (aka kernel)
                   matrix
            :type weights_initializer: string
            :param bias_initializer: Initializer for the bias vector
            :type bias_initializer: string
            :param weights_regularizer: regularizer applied to the weights
                   (kernal) matrix
            :type weights_regularizer: string
            :param bias_regularizer: reguralizer function applied to biase vector.
            :type bias_regularizer: string
            :param activity_regularizer: Regularizer applied to the output of the
                   layer (activation)
            :type activity_regularizer: string
            :param norm: type of normalization to use 'batch' or 'layer'
            :type norm: string, default None
            :param norm_params: parameters to pass to normalization function
            :type norm_params: dictionary
            :param activation: Activation function to use.
            :type activation: string
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: float
        """
        super(DateWave, self).__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  year FCStack')
        self.year_fc = FCStack(
            num_layers=1,
            default_fc_size=1,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=weights_constraint,
            # default_bias_constraint=bias_constraint,
            default_norm=None,
            default_norm_params=None,
            default_activation=None,
            default_dropout=dropout,
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
            :param mask: bool specifying masked values
            :type mask: bool
         """
        # ================ Embeddings ================
        input_vector = tf.cast(inputs, tf.float32)
        scaled_year = self.year_fc(
            input_vector[:, 0:1],
            training=training,
            mask=mask
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
        # logger.debug('  flatten hidden: {0}'.format(hidden))

        hidden = self.fc_stack(
            hidden,
            training=training,
            mask=mask
        )

        return {'encoder_output': hidden}
