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
from ludwig.utils.registry import Registry, register_default
from ludwig.modules.embedding_modules import EmbedSparse
from ludwig.modules.fully_connected_modules import FCStack

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry()


class SetEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register_default(name='embed')
class SetSparseEncoder(SetEncoder):

    def __init__(
            self,
            vocab,
            representation='dense',
            embedding_size=50,
            embeddings_trainable=True,
            pretrained_embeddings=None,
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
            dropout=0.0,
            reduce_output='sum',
            **kwargs
    ):
        super(SetSparseEncoder, self).__init__()
        logger.debug(' {}'.format(self.name))

        logger.debug('  EmbedSparse')
        self.embed_sparse = EmbedSparse(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer,
            reduce_output=reduce_output,
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

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The inputs fed into the encoder.
                   Shape: [batch x 1], type tf.int32

            :param return: embeddings of shape [batch x embed size], type tf.float32
        """
        hidden = self.embed_sparse(inputs, training=training, mask=mask)
        hidden = self.fc_stack(hidden, training=training, mask=mask)

        return hidden
