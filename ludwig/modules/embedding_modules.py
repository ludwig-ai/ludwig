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

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Embedding

from ludwig.constants import TYPE
from ludwig.modules.initializer_modules import get_initializer
from ludwig.utils.data_utils import load_pretrained_embeddings

logger = logging.getLogger(__name__)


def embedding_matrix(
        vocab,
        embedding_size,
        representation='dense',
        embeddings_trainable=True,
        pretrained_embeddings=None,
        force_embedding_size=False,
        embedding_initializer=None,
):
    vocab_size = len(vocab)
    if representation == 'dense':
        if pretrained_embeddings is not None and pretrained_embeddings is not False:
            embeddings_matrix = load_pretrained_embeddings(
                pretrained_embeddings, vocab
            )
            if embeddings_matrix.shape[-1] != embedding_size:
                raise ValueError(
                    'The size of the pretrained embeddings is {}, '
                    'but the specified embedding_size is {}. '
                    'Please change the embedding_size accordingly.'.format(
                        embeddings_matrix.shape[-1],
                        embedding_size
                    ))
            embedding_initializer_obj = tf.constant(embeddings_matrix,
                                                    dtype=tf.float32)

        else:
            if vocab_size < embedding_size and not force_embedding_size:
                logger.info(
                    '  embedding_size ({}) is greater than vocab_size ({}). '
                    'Setting embedding size to be equal to vocab_size.'.format(
                        embedding_size, vocab_size
                    ))
                embedding_size = vocab_size

            if embedding_initializer is not None:
                embedding_initializer_obj_ref = get_initializer(
                    embedding_initializer)
            else:
                embedding_initializer_obj_ref = get_initializer(
                    {TYPE: 'uniform', 'minval': -1.0, 'maxval': 1.0})
            embedding_initializer_obj = embedding_initializer_obj_ref(
                [vocab_size, embedding_size])

        embeddings = tf.Variable(
            embedding_initializer_obj,
            trainable=embeddings_trainable,
            name='embeddings'
        )

    elif representation == 'sparse':
        embedding_size = vocab_size
        embeddings = tf.Variable(
            get_initializer('identity')([vocab_size, embedding_size]),
            trainable=False,
            name='embeddings'
        )

    else:
        raise Exception(
            'Embedding representation {} not supported.'.format(
                representation))

    return embeddings, embedding_size


def embedding_matrix_on_device(
        vocab,
        embedding_size,
        representation='dense',
        embeddings_trainable=True,
        pretrained_embeddings=None,
        force_embedding_size=False,
        embeddings_on_cpu=False,
        embedding_initializer=None
):
    if embeddings_on_cpu:
        with tf.device('/cpu:0'):
            embeddings, embedding_size = embedding_matrix(
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                force_embedding_size=force_embedding_size,
                embedding_initializer=embedding_initializer
            )
    else:
        embeddings, embedding_size = embedding_matrix(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embedding_initializer=embedding_initializer
        )

    # logger.debug('  embeddings: {0}'.format(embeddings))

    return embeddings, embedding_size


class Embed(Layer):
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None
    ):
        super(Embed, self).__init__()
        self.supports_masking = True

        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )

        if embedding_regularizer:
            embedding_regularizer_obj = tf.keras.regularizers.get(
                embedding_regularizer)
            self.add_loss(lambda: embedding_regularizer_obj(self.embeddings))

        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

    def call(self, inputs, training=None, mask=None):
        embedded = tf.nn.embedding_lookup(
            self.embeddings, inputs, name='embeddings_lookup'
        )

        if self.dropout:
            embedded = self.dropout(embedded, training=training)

        return embedded


class EmbedWeighted(Layer):
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None
    ):
        super(EmbedWeighted, self).__init__()

        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )
        self.vocab_length = len(vocab)

        if embedding_regularizer:
            embedding_regularizer_obj = tf.keras.regularizers.get(
                embedding_regularizer)
            self.add_loss(lambda: embedding_regularizer_obj(self.embeddings))

        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

    def call(self, inputs, training=None, mask=None):
        signed_input = tf.cast(tf.sign(tf.abs(inputs)), tf.int32)
        multiple_hot_indexes = tf.multiply(
            signed_input,
            tf.constant(np.array([range(self.vocab_length)], dtype=np.int32))
        )
        embedded = tf.nn.embedding_lookup(
            self.embeddings, multiple_hot_indexes, name='embeddings_lookup'
        )

        # Get the multipliers to embeddings
        weights_mask = tf.expand_dims(inputs, -1)
        weighted_embedded = tf.multiply(embedded, weights_mask)

        embedded_reduced = tf.reduce_sum(weighted_embedded, 1)

        if self.dropout:
            embedded_reduced = self.dropout(embedded_reduced,
                                            training=training)

        return embedded_reduced


class EmbedSparse(Layer):
    def __init__(
            self,
            vocab,
            embedding_size=50,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None,
            reduce_output='sum'
    ):
        super(EmbedSparse, self).__init__()

        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )

        if embedding_regularizer:
            embedding_regularizer_obj = tf.keras.regularizers.get(
                embedding_regularizer)
            self.add_loss(lambda: embedding_regularizer_obj(self.embeddings))

        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

        self.reduce_output = reduce_output

    def call(self, inputs, training=None, mask=None):
        idx = tf.where(tf.equal(inputs, True))

        sparse_multiple_hot_indexes = tf.SparseTensor(
            idx,
            idx[:, 1],
            tf.shape(inputs, out_type=tf.int64)
        )

        embedded_reduced = tf.nn.embedding_lookup_sparse(
            self.embeddings,
            sparse_multiple_hot_indexes,
            sp_weights=None,
            combiner=self.reduce_output
        )

        if self.dropout:
            embedded_reduced = self.dropout(
                embedded_reduced, training=training
            )

        return embedded_reduced


class EmbedSequence(Layer):
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=0.0,
            embedding_initializer=None,
            embedding_regularizer=None
    ):
        super(EmbedSequence, self).__init__()
        self.supports_masking = True

        self.embeddings, self.embedding_size = embedding_matrix_on_device(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            embedding_initializer=embedding_initializer,
        )

        if embedding_regularizer:
            embedding_regularizer_obj = tf.keras.regularizers.get(
                embedding_regularizer)
            self.add_loss(lambda: embedding_regularizer_obj(self.embeddings))

        if dropout > 0:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

    def call(self, inputs, training=None, mask=None):
        embedded = tf.nn.embedding_lookup(
            self.embeddings, inputs, name='embeddings_lookup'
        )

        if mask is not None:
            mask_matrix = tf.cast(
                tf.expand_dims(mask, -1),
                dtype=tf.float32
            )
            embedded = tf.multiply(embedded, mask_matrix)

        if self.dropout:
            embedded = self.dropout(embedded, training=training)

        return embedded


class TokenAndPositionEmbedding(Layer):
    def __init__(self,
                 max_length,
                 vocab,
                 embedding_size,
                 representation='dense',
                 embeddings_trainable=True,
                 pretrained_embeddings=None,
                 force_embedding_size=False,
                 embeddings_on_cpu=False,
                 dropout=0.0,
                 embedding_initializer=None,
                 embedding_regularizer=None
                 ):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embed = EmbedSequence(
            vocab=vocab,
            embedding_size=embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer
        )
        self.position_embed = Embedding(
            input_dim=max_length,
            output_dim=self.token_embed.embedding_size
        )

    def call(self, inputs, training=None, mask=None):
        max_length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=max_length, delta=1)
        positions_hidden = self.position_embed(positions)
        token_hidden = self.token_embed(inputs)
        return token_hidden + positions_hidden
