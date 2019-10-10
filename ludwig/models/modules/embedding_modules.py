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

from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.utils.data_utils import load_pretrained_embeddings

logger = logging.getLogger(__name__)


def embedding_matrix(
        vocab,
        embedding_size,
        representation='dense',
        embeddings_trainable=True,
        pretrained_embeddings=None,
        force_embedding_size=False,
        initializer=None,
        regularizer=None
):
    vocab_size = len(vocab)
    if representation == 'dense':
        if pretrained_embeddings is not None and pretrained_embeddings is not False:
            embeddings_matrix = load_pretrained_embeddings(
                pretrained_embeddings, vocab)
            if embeddings_matrix.shape[-1] != embedding_size:
                raise ValueError(
                    'The size of the pretrained embeddings is {}, '
                    'but the specified embedding_size is {}. '
                    'Please change the embedding_size accordingly.'.format(
                        embeddings_matrix.shape[-1],
                        embedding_size
                    ))
            initializer_obj = tf.constant(embeddings_matrix, dtype=tf.float32)
        else:
            if vocab_size < embedding_size and not force_embedding_size:
                logger.info(
                    '  embedding_size ({}) is greater than vocab_size ({}). '
                    'Setting embedding size to be equal to vocab_size.'.format(
                        embedding_size, vocab_size
                    ))
                embedding_size = vocab_size

            if initializer is not None:
                initializer_obj_ref = get_initializer(initializer)
            else:
                initializer_obj_ref = get_initializer(
                    {'type': 'uniform', 'minval': -1.0, 'maxval': 1.0})
            initializer_obj = initializer_obj_ref([vocab_size, embedding_size])

        embeddings = tf.compat.v1.get_variable('embeddings',
                                     initializer=initializer_obj,
                                     trainable=embeddings_trainable,
                                     regularizer=regularizer)

    elif representation == 'sparse':
        embedding_size = vocab_size
        embeddings = tf.compat.v1.get_variable('embeddings',
                                     initializer=get_initializer('identity')(
                                         [vocab_size, embedding_size]),
                                     trainable=False)

    else:
        raise Exception(
            'Embedding representation {} not supported.'.format(representation))

    return embeddings, embedding_size


class Embed:
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=False,
            initializer=None,
            regularize=True
    ):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.representation = representation
        self.embeddings_trainable = embeddings_trainable
        self.pretrained_embeddings = pretrained_embeddings
        self.force_embedding_size = force_embedding_size
        self.embeddings_on_cpu = embeddings_on_cpu
        self.dropout = dropout
        self.initializer = initializer
        self.regularize = regularize

    def __call__(
            self,
            input_ids,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        if not self.regularize:
            regularizer = None

        if self.embeddings_on_cpu:
            with tf.device('/cpu:0'):
                embeddings, embedding_size = embedding_matrix(
                    self.vocab,
                    self.embedding_size,
                    representation=self.representation,
                    embeddings_trainable=self.embeddings_trainable,
                    pretrained_embeddings=self.pretrained_embeddings,
                    force_embedding_size=self.force_embedding_size,
                    initializer=self.initializer,
                    regularizer=regularizer
                )
        else:
            embeddings, embedding_size = embedding_matrix(
                self.vocab,
                self.embedding_size,
                representation=self.representation,
                embeddings_trainable=self.embeddings_trainable,
                pretrained_embeddings=self.pretrained_embeddings,
                force_embedding_size=self.force_embedding_size,
                initializer=self.initializer,
                regularizer=regularizer
            )
        logger.debug('  embeddings: {0}'.format(embeddings))

        embedded = tf.nn.embedding_lookup(embeddings, input_ids,
                                          name='embeddings_lookup')
        logger.debug('  embedded: {0}'.format(embedded))

        if self.dropout and dropout_rate is not None:
            embedded = tf.layers.dropout(embedded, rate=dropout_rate,
                                         training=is_training)
            logger.debug('  embedded_dropout: {}'.format(embedded))

        return embedded, embedding_size


class EmbedWeighted:
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            dropout=False,
            initializer=None,
            regularize=True
    ):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.representation = representation
        self.embeddings_trainable = embeddings_trainable
        self.pretrained_embeddings = pretrained_embeddings
        self.force_embedding_size = force_embedding_size
        self.embeddings_on_cpu = embeddings_on_cpu
        self.dropout = dropout
        self.initializer = initializer
        self.regularize = regularize

    def __call__(
            self,
            input_ids,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        if not self.regularize:
            regularizer = None

        if self.embeddings_on_cpu:
            with tf.device('/cpu:0'):
                embeddings, embedding_size = embedding_matrix(
                    self.vocab,
                    self.embedding_size,
                    representation=self.representation,
                    embeddings_trainable=self.embeddings_trainable,
                    pretrained_embeddings=self.pretrained_embeddings,
                    force_embedding_size=self.force_embedding_size,
                    initializer=self.initializer,
                    regularizer=regularizer
                )
        else:
            embeddings, embedding_size = embedding_matrix(
                self.vocab,
                self.embedding_size,
                representation=self.representation,
                embeddings_trainable=self.embeddings_trainable,
                pretrained_embeddings=self.pretrained_embeddings,
                force_embedding_size=self.force_embedding_size,
                initializer=self.initializer,
                regularizer=regularizer
            )
        logger.debug('  embeddings: {0}'.format(embeddings))

        signed_input = tf.cast(tf.sign(tf.abs(input_ids)), tf.int32)
        multiple_hot_indexes = tf.multiply(
            signed_input,
            tf.constant(np.array([range(len(self.vocab))], dtype=np.int32))
        )
        embedded = tf.nn.embedding_lookup(
            embeddings,
            multiple_hot_indexes,
            name='embeddings_lookup'
        )
        logger.debug('  embedded: {0}'.format(embedded))

        # Get the multipliers to embeddings
        weights_mask = tf.expand_dims(input_ids, -1)
        weighted_embedded = tf.multiply(embedded, weights_mask)
        logger.debug('  weighted_embedded: {0}'.format(weighted_embedded))

        embedded_reduced = tf.reduce_sum(weighted_embedded, 1)
        logger.debug('  embedded_reduced: {0}'.format(embedded_reduced))

        if self.dropout and dropout_rate is not None:
            embedded = tf.layers.dropout(embedded, rate=dropout_rate,
                                         training=is_training)
            logger.debug('  embedded_dropout: {}'.format(embedded))

        return embedded_reduced, embedding_size


class EmbedSparse:
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            reduce_output='sum',
            dropout=False,
            initializer=None,
            regularize=True
    ):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.representation = representation
        self.embeddings_trainable = embeddings_trainable
        self.pretrained_embeddings = pretrained_embeddings
        self.force_embedding_size = force_embedding_size
        self.embeddings_on_cpu = embeddings_on_cpu
        self.reduce_output = reduce_output
        self.dropout = dropout
        self.initializer = initializer
        self.regularize = regularize

    def __call__(
            self,
            input_sparse,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        if not self.regularize:
            regularizer = None

        if self.embeddings_on_cpu:
            with tf.device('/cpu:0'):
                embeddings, embedding_size = embedding_matrix(
                    self.vocab,
                    self.embedding_size,
                    representation=self.representation,
                    embeddings_trainable=self.embeddings_trainable,
                    pretrained_embeddings=self.pretrained_embeddings,
                    force_embedding_size=self.force_embedding_size,
                    initializer=self.initializer,
                    regularizer=regularizer
                )
        else:
            embeddings, embedding_size = embedding_matrix(
                self.vocab,
                self.embedding_size,
                representation=self.representation,
                embeddings_trainable=self.embeddings_trainable,
                pretrained_embeddings=self.pretrained_embeddings,
                force_embedding_size=self.force_embedding_size,
                initializer=self.initializer,
                regularizer=regularizer
            )
        logger.debug('  embeddings: {0}'.format(embeddings))

        multiple_hot_indexes = tf.multiply(
            input_sparse,
            tf.constant(np.array([range(len(self.vocab))], dtype=np.int32))
        )

        idx = tf.where(tf.not_equal(multiple_hot_indexes, 0))

        sparse_multiple_hot_indexes = tf.SparseTensor(
            idx,
            tf.gather_nd(multiple_hot_indexes, idx),
            tf.shape(multiple_hot_indexes, out_type=tf.int64)
        )

        embedded_reduced = tf.nn.embedding_lookup_sparse(
            embeddings,
            sparse_multiple_hot_indexes,
            sp_weights=None,
            combiner=self.reduce_output
        )
        logger.debug('  embedded_reduced: {0}'.format(embedded_reduced))

        # Old dense implementation
        # embedded = tf.nn.embedding_lookup(
        #     feature_embeddings,
        #     multiple_hot_indexes,
        #     name=input_feature['name'] + '_embeddings_lookup',
        # )
        # mask = tf.cast(tf.expand_dims(tf.sign(tf.abs(multiple_hot_indexes)), -1), tf.float32)
        # masked_embedded = tf.multiply(embedded, mask)
        # embedded_reduced = tf.reduce_sum(masked_embedded, 1)

        if self.dropout and dropout_rate is not None:
            embedded_reduced = tf.layers.dropout(embedded_reduced,
                                                 rate=dropout_rate,
                                                 training=is_training)
            logger.debug(
                '  embedded_reduced_dropout: {}'.format(embedded_reduced))

        return embedded_reduced, embedding_size


class EmbedSequence:
    def __init__(
            self,
            vocab,
            embedding_size,
            representation='dense',
            embeddings_trainable=True,
            pretrained_embeddings=None,
            force_embedding_size=False,
            embeddings_on_cpu=False,
            mask=True,
            dropout=False,
            initializer=None,
            regularize=True
    ):
        self.embed = Embed(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            force_embedding_size=force_embedding_size,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

        self.mask = mask

    def __call__(
            self,
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        embedded, embedding_size = self.embed(
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
        )

        if self.mask:
            mask_matrix = tf.cast(
                tf.expand_dims(tf.sign(tf.abs(input_sequence)), -1),
                dtype=tf.float32
            )
            embedded = tf.multiply(embedded, mask_matrix)

        return embedded, embedding_size
