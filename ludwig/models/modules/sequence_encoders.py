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

from ludwig.models.modules.convolutional_modules import ConvStack1D, \
    StackParallelConv1D, ParallelConv1D
from ludwig.models.modules.embedding_modules import EmbedSequence
from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.recurrent_modules import RecurrentStack
from ludwig.models.modules.recurrent_modules import reduce_sequence


class EmbedEncoder:

    def __init__(
            self,
            vocab,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            initializer=None,
            dropout=False,
            regularize=True,
            reduce_output='sum',
            **kwargs
    ):
        self.reduce_output = reduce_output

        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

    def __call__(
            self,
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        # ================ Embeddings ================
        embedded_sequence, embedding_size = self.embed_sequence(
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
        )

        hidden = reduce_sequence(embedded_sequence, self.reduce_output)

        return hidden, embedding_size


class ParallelCNN(object):

    def __init__(
            self,
            should_embed=True,
            vocab=None,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            conv_layers=None,
            num_conv_layers=None,
            filter_size=3,
            num_filters=256,
            pool_size=None,
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
            norm=None,
            dropout=False,
            activation='relu',
            initializer=None,
            regularize=True,
            reduce_output='max',
            **kwargs):
        """
            :param input_sequence: The input sequence fed into the parallel cnn
            :type input_sequence:
            :param regularizer: The method of regularization that is being
            :type regularizer:
            :param dropout_rate: Probability of dropping a neuron in a layer
            :type dropout_rate: Float
            :param vocab: Vocabulary in the dataset
            :type vocab: List
            :param representation: Either dense or sparse representations
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: The dimension of the embedding that has been chosen
            :type embedding_size: Integer
            :param filter_sizes: Size of the filter used in the convolutions
            :type filter_sizes: Tuple (Integer)
            :param num_filters: Number of filters to apply on the input for a given filter size
            :type num_filters: Tuple (Integer)
            :param fc_sizes: Fully connected dimensions at the end of the convolution layers
            :type fc_sizes: Tuple (Integer)
            :param norms: TODO
            :type norms:
            :param activations: Type of activation function being used in the model
            :type activations: Str
            :param regularize: TODO
            :type regularize:
            :param embeddings_trainable: Argument that determines if the embeddings in the model are trainable end to end
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: Represents whether the embedd
            :type pretrained_embeddings: Boolean
            :param embeddings_on_cpu: TODO: clarify (Whether the embeddings should be trained on the CPU)
            :type embeddings_on_cpu: Boolean
            :param should_embed: Represents a boolean value determining if there is a need to embed the input sequence
            :type should_embed: Boolean
            :param is_training: Whether this is training or not
            :type is_training: Boolean
            :returns: hidden, hidden_size - the hidden layer and hidden size
            """

        self.should_embed = should_embed

        if conv_layers is not None and num_conv_layers is None:
            # use custom-defined layers
            self.conv_layers = conv_layers
            self.num_conv_layers = len(conv_layers)
        elif conv_layers is None and num_conv_layers is not None:
            # generate num_conv_layers with default parameters
            self.conv_layers = None
            self.num_conv_layers = num_conv_layers
        elif conv_layers is None and num_conv_layers is None:
            # use default layers with varying filter sizes
            self.conv_layers = [
                {'filter_size': 2},
                {'filter_size': 3},
                {'filter_size': 4},
                {'filter_size': 5}
            ]
            self.num_conv_layers = 4
        else:
            raise ValueError(
                'Invalid layer parametrization, use either conv_layers or num_conv_layers')

        if fc_layers is not None and num_fc_layers is None:
            # use custom-defined layers
            fc_layers = fc_layers
            num_fc_layers = len(fc_layers)
        elif fc_layers is None and num_fc_layers is not None:
            # generate num_fc_layers with default parameters
            fc_layers = None
            num_fc_layers = num_fc_layers
        elif fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [
                {'fc_size': 512},
                {'fc_size': 256}
            ]
            num_fc_layers = 2
        else:
            raise ValueError(
                'Invalid layer parametrization, use either fc_layers or num_fc_layers')

        self.reduce_output = reduce_output

        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

        self.parallel_conv_1d = ParallelConv1D(
            layers=self.conv_layers,
            default_filter_size=filter_size,
            default_num_filters=num_filters,
            default_pool_size=pool_size,
            default_activation=activation,
            default_norm=norm,
            default_dropout=dropout,
            default_initializer=initializer,
            default_regularize=regularize
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
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        # ================ Embeddings ================
        if self.should_embed:
            embedded_input_sequence, embedding_size = self.embed_sequence(
                input_sequence,
                regularizer,
                dropout_rate,
                is_training=True
            )
        else:
            embedded_input_sequence = input_sequence
            while len(embedded_input_sequence.shape) < 3:
                embedded_input_sequence = tf.expand_dims(
                    embedded_input_sequence, -1)
            embedding_size = 1

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_input_sequence
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ Conv Layers ================
        hidden = self.parallel_conv_1d(
            hidden,
            embedding_size,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )
        hidden_size = sum(
            [conv_layer['num_filters'] for conv_layer in self.conv_layers]
        )
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = reduce_sequence(hidden, self.reduce_output)

            # ================ FC Layers ================
            hidden_size = hidden.shape.as_list()[-1]
            logging.debug('  flatten hidden: {0}'.format(hidden))

            hidden = self.fc_stack(
                hidden,
                hidden_size,
                regularizer=regularizer,
                dropout_rate=dropout_rate,
                is_training=is_training
            )
            hidden_size = hidden.shape.as_list()[-1]

        return hidden, hidden_size


class StackedCNN:

    def __init__(
            self,
            should_embed=True,
            vocab=None,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            conv_layers=None,
            num_conv_layers=None,
            filter_size=5,
            num_filters=256,
            pool_size=None,
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
            norm=None,
            activation='relu',
            dropout=False,
            initializer=None,
            regularize=True,
            reduce_output='max',
            **kwargs
    ):
        """
            :param input_sequence: The input sequence fed into the stacked cnn
            :type input_sequence:
            :param regularizer: The method of regularization that is being used
            :type regularizer:
            :param dropout_rate: Probability of dropping a neuron in a layer
            :type dropout_rate: Float
            :param vocab: Vocabulary in the dataset
            :type vocab: List
            :param representation: Either dense or sparse representations
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: The dimension of the embedding that has been chosen
            :type embedding_size: Integer
            :param filter_sizes: Size of the filter used in the convolutions
            :type filter_sizes: Tuple (Integer)
            :param num_filters: Number of filters to apply on the input for a given filter size
            :type num_filters: Tuple (Integer)
            :param pool_sizes: Use the pooling of features in Convlutional Neural Nets TODO
            :type pool_sizes: Integer or None
            :param fc_sizes: Fully connected dimensions at the end of the convolution layers
            :param activations: Type of activation function being used in the model
            :type activations: Str
            :param regularize: TODO
            :type regularize:
            :type fc_sizes: Tuple (Integer)
            :param norms: TODO
            :type norms:
            :param should_embed: Represents a boolean value determining if there is a need to embed the input sequence
            :type should_embed: Boolean
            :param embeddings_trainable: Argument that determines if the embeddings in the model are trainable end to end
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: Represents whether the embedd
            :type pretrained_embeddings: Boolean
            :param embeddings_on_cpu: TODO: clarify (Whether the embeddings should be trained on the CPU)
            :type embeddings_on_cpu: Boolean
            :param is_training: Whether this is training or not
            :type is_training: Boolean
            :returns: hidden, hidden_size - the hidden layer and hidden size
        """

        if conv_layers is not None and num_conv_layers is None:
            # use custom-defined layers
            self.conv_layers = conv_layers
            self.num_conv_layers = len(conv_layers)
        elif conv_layers is None and num_conv_layers is not None:
            # generate num_conv_layers with default parameters
            self.conv_layers = None
            self.num_conv_layers = num_conv_layers
        elif conv_layers is None and num_conv_layers is None:
            # use default layers with varying filter sizes
            self.conv_layers = [
                {
                    'filter_size': 7,
                    'pool_size': 3,
                    'regularize': False
                },
                {
                    'filter_size': 7,
                    'pool_size': 3,
                    'regularize': False
                },
                {
                    'filter_size': 3,
                    'pool_size': None,
                    'regularize': False
                },
                {
                    'filter_size': 3,
                    'pool_size': None,
                    'regularize': False
                },
                {
                    'filter_size': 3,
                    'pool_size': None,
                    'regularize': True
                },
                {
                    'filter_size': 3,
                    'pool_size': 3,
                    'regularize': True
                }
            ]
            self.num_conv_layers = 6
        else:
            raise ValueError(
                'Invalid layer parametrization, use either conv_layers or num_conv_layers')

        if fc_layers is not None and num_fc_layers is None:
            # use custom-defined layers
            fc_layers = fc_layers
            num_fc_layers = len(fc_layers)
        elif fc_layers is None and num_fc_layers is not None:
            # generate num_fc_layers with default parameters
            fc_layers = None
            num_fc_layers = num_fc_layers
        elif fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [
                {'fc_size': 512},
                {'fc_size': 256}
            ]
            num_fc_layers = 2
        else:
            raise ValueError(
                'Invalid layer parametrization, use either fc_layers or num_fc_layers')

        self.should_embed = should_embed
        self.reduce_output = reduce_output

        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

        self.conv_stack_1d = ConvStack1D(
            layers=self.conv_layers,
            default_filter_size=filter_size,
            default_num_filters=num_filters,
            default_activation=activation,
            default_norm=norm,
            default_pool_size=pool_size,
            default_dropout=dropout,
            default_initializer=initializer,
            default_regularize=regularize
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
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        # ================ Embeddings ================
        if self.should_embed:
            embedded_input_sequence, self.embedding_size = self.embed_sequence(
                input_sequence,
                regularizer,
                dropout_rate,
                is_training=True
            )
        else:
            embedded_input_sequence = input_sequence
            while len(embedded_input_sequence.shape) < 3:
                embedded_input_sequence = tf.expand_dims(
                    embedded_input_sequence, -1)
            self.embedding_size = embedded_input_sequence.shape[-1]

        hidden = embedded_input_sequence
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ Conv Layers ================
        with tf.variable_scope('stack_conv'):
            hidden = self.conv_stack_1d(
                hidden,
                self.embedding_size,
                regularizer=regularizer,
                dropout_rate=dropout_rate,
                is_training=is_training
            )
        hidden_size = self.conv_layers[-1]['num_filters']
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = reduce_sequence(hidden, self.reduce_output)

            # ================ FC Layers ================
            hidden_size = hidden.shape.as_list()[-1]
            logging.debug('  flatten hidden: {0}'.format(hidden))

            hidden = self.fc_stack(
                hidden,
                hidden_size,
                regularizer=regularizer,
                dropout_rate=dropout_rate,
                is_training=is_training
            )
            hidden_size = hidden.shape.as_list()[-1]

        return hidden, hidden_size


class StackedParallelCNN:

    def __init__(
            self,
            should_embed=True,
            vocab=None,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            stacked_layers=None,
            num_stacked_layers=None,
            filter_size=3,
            num_filters=256,
            stride=1,
            pool_size=None,
            pool_stride=1,
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
            activation='relu',
            norm=None,
            dropout=False,
            initializer=None,
            regularize=True,
            reduce_output='max',
            **kwargs
    ):
        """
            :param input_sequence: The input sequence fed into the stacked parallel cnn
            :type input_sequence:
            :param regularizer: The method of regularization that is being
            :type regularizer:
            :param dropout: Probability of dropping a neuron in a layer
            :type dropout: Float
            :param vocab: Vocabulary in the dataset
            :type vocab: List
            :param representation: Either dense or sparse representations
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: The dimension of the embedding that has been chosen
            :type embedding_size: Integer
            :param filter_sizes: Size of the filter used in the convolutions
            :type filter_sizes: Tuple(Tuple(Integer))
            :param num_filters: Number of filters to apply on the input for a given filter size
            :type num_filters: Tuple(Tuple(Integer))
            :param fc_sizes: Fully connected dimensions at the end of the convolution layers
            :type fc_sizes: Tuple (Integer)
            :param activations: Type of activation function being used in the model
            :type activations: Str
            :param regularize: TODO
            :type regularize:
            :param norms: TODO
            :type norms:
            :param should_embed: Represents a boolean value determining if there is a need to embed the input sequence
            :type should_embed: Boolean
            :param embeddings_trainable: Argument that determines if the embeddings in the model are trainable end to end
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: Represents whether the embedd
            :type pretrained_embeddings: Boolean
            :param embeddings_on_cpu: TODO: clarify (Whether the embeddings should be trained on the CPU)
            :type embeddings_on_cpu: Boolean
            :param is_training: Whether this is training or not
            :type is_training: Boolean
            :returns: hidden, hidden_size - the hidden layer and hidden size
        """
        if stacked_layers is not None and num_stacked_layers is None:
            # use custom-defined layers
            self.stacked_layers = stacked_layers
            self.num_stacked_layers = len(stacked_layers)
        elif stacked_layers is None and num_stacked_layers is not None:
            # generate num_conv_layers with default parameters
            self.stacked_layers = None
            self.num_stacked_layers = num_stacked_layers
        elif stacked_layers is None and num_stacked_layers is None:
            # use default layers with varying filter sizes
            self.stacked_layers = [
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ],
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ],
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ]
            ]
            self.num_stacked_layers = 6
        else:
            raise ValueError(
                'Invalid layer parametrization, use either stacked_layers or num_stacked_layers')

        if fc_layers is not None and num_fc_layers is None:
            # use custom-defined layers
            fc_layers = fc_layers
            num_fc_layers = len(fc_layers)
        elif fc_layers is None and num_fc_layers is not None:
            # generate num_fc_layers with default parameters
            fc_layers = None
            num_fc_layers = num_fc_layers
        elif fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [
                {'fc_size': 512},
                {'fc_size': 256}
            ]
            num_fc_layers = 2
        else:
            raise ValueError(
                'Invalid layer parametrization, use either fc_layers or num_fc_layers')

        self.should_embed = should_embed
        self.reduce_output = reduce_output

        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

        self.stack_parallel_conv_1d = StackParallelConv1D(
            stacked_layers=self.stacked_layers,
            default_filter_size=filter_size,
            default_num_filters=num_filters,
            default_pool_size=pool_size,
            default_activation=activation,
            default_norm=norm,
            default_stride=stride,
            default_pool_stride=pool_stride,
            default_dropout=dropout,
            default_initializer=initializer,
            default_regularize=regularize
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
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        # ================ Embeddings ================
        if self.should_embed:
            embedded_input_sequence, self.embedding_size = self.embed_sequence(
                input_sequence,
                regularizer,
                dropout_rate,
                is_training=True
            )
        else:
            embedded_input_sequence = input_sequence
            while len(embedded_input_sequence.shape) < 3:
                embedded_input_sequence = tf.expand_dims(
                    embedded_input_sequence,
                    -1
                )
            self.embedding_size = embedded_input_sequence.shape[-1]

        hidden = embedded_input_sequence
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ Conv Layers ================
        with tf.variable_scope('stack_parallel_conv'):
            hidden = self.stack_parallel_conv_1d(
                hidden,
                self.embedding_size,
                regularizer=regularizer,
                dropout_rate=dropout_rate,
                is_training=is_training
            )
        hidden_size = 0
        for stack in self.stacked_layers:
            hidden_size += stack[-1]['num_filters']
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = reduce_sequence(hidden, self.reduce_output)

            # ================ FC Layers ================
            hidden_size = hidden.shape.as_list()[-1]
            logging.debug('  flatten hidden: {0}'.format(hidden))

            hidden = self.fc_stack(
                hidden,
                hidden_size,
                regularizer=regularizer,
                dropout_rate=dropout_rate,
                is_training=is_training
            )
            hidden_size = hidden.shape.as_list()[-1]

        return hidden, hidden_size


class RNN:

    def __init__(
            self,
            should_embed=True,
            vocab=None,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            num_layers=1,
            state_size=256,
            cell_type='rnn',
            bidirectional=False,
            dropout=False,
            initializer=None,
            regularize=True,
            reduce_output='last',
            **kwargs
    ):
        """
            :param input_sequence: The input sequence fed into the rnn
            :type input_sequence:
            :param dropout_rate: Probability of dropping a neuron in a layer
            :type dropout_rate: Float
            :param vocab: Vocabulary in the dataset
            :type vocab: List
            :param representation: Either dense or sparse representations
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: The dimension of the embedding that has been chosen
            :type embedding_size: Integer
            :param state_size: Size of the hidden state TODO: Confirm
            :type state_size: Integer
            :param cell_type: The type of cell being used (e.g. 'rnn')
            :type: Str
            :param num_layers: Number of recurrent layers
            :type num_layers: Integer
            :param bidirectional: Using Bidirectional RNN's
            :type bidirectional: Boolean
            :param reduce_output: TODO
            :type reduce_output:
            :param should_embed: Represents a boolean value determining if there is a need to embed the input sequence
            :type should_embed: Boolean
            :param embeddings_trainable: Argument that determines if the embeddings in the model are trainable end to end
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: Represents whether the embedd
            :type pretrained_embeddings: Boolean
            :param embeddings_on_cpu: TODO: clarify (Whether the embeddings should be trained on the CPU)
            :type embeddings_on_cpu: Boolean
            :param is_training: Whether this is training or not
            :type is_training: Boolean
            :returns: hidden, hidden_size - the hidden layer and hidden size
        """

        self.should_embed = should_embed

        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
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
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        # ================ Embeddings ================
        if self.should_embed:
            embedded_input_sequence, self.embedding_size = self.embed_sequence(
                input_sequence,
                regularizer,
                dropout_rate,
                is_training=True
            )
        else:
            embedded_input_sequence = input_sequence
            while len(embedded_input_sequence.shape) < 3:
                embedded_input_sequence = tf.expand_dims(
                    embedded_input_sequence,
                    -1
                )
            self.embedding_size = embedded_input_sequence.shape[-1]
        logging.debug('  hidden: {0}'.format(embedded_input_sequence))

        # ================ RNN ================
        hidden, hidden_size = self.recurrent_stack(
            embedded_input_sequence,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )

        return hidden, hidden_size


class CNNRNN:

    def __init__(
            self,
            should_embed=True,
            vocab=None,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            conv_layers=None,
            num_conv_layers=None,
            filter_size=5,
            num_filters=256,
            norm=None,
            activation='relu',
            pool_size=None,
            num_rec_layers=1,
            state_size=256,
            cell_type='rnn',
            bidirectional=False,
            dropout=False,
            initializer=None,
            regularize=True,
            reduce_output='last',
            **kwargs
    ):
        if conv_layers is not None and num_conv_layers is None:
            # use custom-defined layers
            self.conv_layers = conv_layers
            self.num_conv_layers = len(conv_layers)
        elif conv_layers is None and num_conv_layers is not None:
            # generate num_conv_layers with default parameters
            self.conv_layers = None
            self.num_conv_layers = num_conv_layers
        elif conv_layers is None and num_conv_layers is None:
            # use default layers with varying filter sizes
            self.conv_layers = [
                {'pool_size': 3},
                {'pool_size': None}
            ]
            self.num_conv_layers = 2
        else:
            raise ValueError(
                'Invalid layer parametrization, use either conv_layers or num_conv_layers')

        self.should_embed = should_embed

        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            initializer=initializer,
            regularize=regularize
        )

        self.conv_stack_1d = ConvStack1D(
            layers=self.conv_layers,
            default_filter_size=filter_size,
            default_num_filters=num_filters,
            default_activation=activation,
            default_norm=norm,
            default_pool_size=pool_size,
            default_dropout=dropout,
            default_initializer=initializer,
            default_regularize=regularize
        )

        self.recurrent_stack = RecurrentStack(
            state_size=state_size,
            cell_type=cell_type,
            num_layers=num_rec_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            regularize=regularize,
            reduce_output=reduce_output
        )

    def __call__(
            self,
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        # ================ Embeddings ================
        if self.should_embed:
            embedded_input_sequence, self.embedding_size = self.embed_sequence(
                input_sequence,
                regularizer,
                dropout_rate,
                is_training=True
            )
        else:
            embedded_input_sequence = input_sequence
            while len(embedded_input_sequence.shape) < 3:
                embedded_input_sequence = tf.expand_dims(
                    embedded_input_sequence,
                    -1
                )
            self.embedding_size = embedded_input_sequence.shape[-1]

        hidden = embedded_input_sequence
        # shape=(?, sequence_length, embedding_size)
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ CNN ================
        hidden = self.conv_stack_1d(
            hidden,
            self.embedding_size,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )
        logging.debug('  hidden: {0}'.format(hidden))

        # ================ RNN ================
        hidden, hidden_size = self.recurrent_stack(
            hidden,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )

        return hidden, hidden_size
