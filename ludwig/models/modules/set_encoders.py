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

from tensorflow.keras.layers import Layer

from ludwig.models.modules.embedding_modules import EmbedSparse

logger = logging.getLogger(__name__)


class SetSparseEncoder(Layer):

    def __init__(
            self,
            vocab=None,
            embedding_size=50,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            dropout_rate=0.0,
            initializer=None,
            regularizer=None,
            **kwargs
    ):
        super(SetSparseEncoder, self).__init__()

        self.embed_sparse = EmbedSparse(
            vocab,
            embedding_size,
            representation='sparse',
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout_rate=dropout_rate,
            initializer=initializer,
            regularizer=regularizer
        )

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """
        embedded = self.embed_sparse(
            inputs, training=None, mask=None
        )
        return embedded
