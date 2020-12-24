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

from ludwig.encoders.base import Encoder
from ludwig.utils.registry import Registry, register, DEFAULT_KEYS
from ludwig.encoders.generic_encoders import PassthroughEncoder
from ludwig.modules.embedding_modules import Embed

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry({
    key: PassthroughEncoder for key in DEFAULT_KEYS + ['passthrough']
})


class CategoricalEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register(name='dense')
class CategoricalEmbedEncoder(CategoricalEncoder):

    def __init__(
            self,
            vocab,
            embedding_size=50,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None,
            **kwargs
    ):
        super(CategoricalEmbedEncoder, self).__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  Embed')
        self.embed = Embed(
            vocab=vocab,
            embedding_size=embedding_size,
            representation='dense',
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer
        )

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """
        embedded = self.embed(
            inputs, training=training, mask=mask
        )
        return embedded


@register(name='sparse')
class CategoricalSparseEncoder(CategoricalEncoder):

    def __init__(
            self,
            vocab,
            embedding_size=50,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None,
            **kwargs
    ):
        super(CategoricalSparseEncoder, self).__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  Embed')
        self.embed = Embed(
            vocab=vocab,
            embedding_size=embedding_size,
            representation='sparse',
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer
        )

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """
        embedded = self.embed(
            inputs, training=training, mask=mask
        )
        return embedded
