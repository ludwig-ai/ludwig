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
            **kwargs
    ):
        super(CategoricalEmbedEncoder, self).__init__()

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """
        return tf.cast(inputs, dtype=tf.float32)

class CategoricalSparseEncoder(Layer):

    def __init__(
            self,
            **kwargs
    ):
        super(CategoricalSparseEncoder, self).__init__()

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32
            :param return: one-hot encoding, shape [batch x number classes], type tf.int32
        """
        return inputs
