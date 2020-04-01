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
from tensorflow.keras.layers import Layer
from ludwig.models.modules.embedding_modules import Embed

logger = logging.getLogger(__name__)


class CategoricalPassthroughEncoder(Layer):

    def __init__(
            self,
            **kwargs
    ):
        super(CategoricalPassthroughEncoder, self).__init__()

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32
        """
        return inputs


class CategoricalEmbedEncoder(Layer):

    def __init__(
            self,
            vocab=None,
            embedding_size=None,
            embeddings_trainable=None,
            pretrained_embeddings=None,
            embeddings_on_cpu=None,
            dropout=None,
            initializer=None,
            regularize=None,
            **kwargs
    ):
        super(CategoricalEmbedEncoder, self).__init__()

        self.vocab = vocab
        self.embedding_size = embedding_size
        self.embeddings_trainable = embeddings_trainable
        self.pretrained_embeddings = pretrained_embeddings
        self.embeddings_on_cpu = embeddings_on_cpu
        self.dropout = dropout
        self.initializer = initializer
        self.regularize = regularize


        self.embed = Embed(
            vocab=self.vocab,
            embedding_size=self.embedding_size,
            representation='dense',
            embeddings_trainable=self.embeddings_trainable,
            pretrained_embeddings=self.pretrained_embeddings,
            embeddings_on_cpu=self.embeddings_on_cpu,
            dropout=self.dropout,
            initializer=self.initializer,
            regularize=self.regularize
        )


    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """

        embedded, embedding_size = self.embed(
            tf.cast(inputs, dtype=tf.int32),
            None,   # todo tf2 need regularizer
            self.dropout,
            self.embeddings_trainable
        )

        # todo tf2: remove tf.squeeze() after Embed() update to return correct
        #           dimension
        embedded = tf.squeeze(embedded)

        return embedded

class CategoricalSparseEncoder(Layer):

    def __init__(
            self,
            vocab=None,
            embedding_size=None,
            embeddings_trainable=None,
            pretrained_embeddings=None,
            embeddings_on_cpu=None,
            dropout=None,
            initializer=None,
            regularize=None,
            **kwargs
    ):
        super(CategoricalSparseEncoder, self).__init__()

        self.vocab = vocab
        self.embedding_size = embedding_size
        self.embeddings_trainable = embeddings_trainable
        self.pretrained_embeddings = pretrained_embeddings
        self.embeddings_on_cpu = embeddings_on_cpu
        self.dropout = dropout
        self.initializer = initializer
        self.regularize = regularize


        self.embed = Embed(
            vocab=self.vocab,
            embedding_size=self.embedding_size,
            representation='sparse',
            embeddings_trainable=self.embeddings_trainable,
            pretrained_embeddings=self.pretrained_embeddings,
            embeddings_on_cpu=self.embeddings_on_cpu,
            dropout=self.dropout,
            initializer=self.initializer,
            regularize=self.regularize
        )


    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """

        embedded, embedding_size = self.embed(
            tf.cast(inputs, dtype=tf.int32),
            None,   # todo tf2 need regularizer
            self.dropout,
            self.embeddings_trainable
        )

        # todo tf2: remove tf.squeeze() after Embed() update to return correct
        #           dimension
        embedded = tf.squeeze(embedded)

        return embedded
