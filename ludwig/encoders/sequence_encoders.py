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

import tensorflow as tf
from tensorflow.keras.layers import Dense

from ludwig.encoders.base import Encoder
from ludwig.utils.registry import Registry, register, register_default
from ludwig.modules.attention_modules import TrasformerStack
from ludwig.modules.convolutional_modules import Conv1DStack, \
    ParallelConv1DStack, ParallelConv1D
from ludwig.modules.embedding_modules import EmbedSequence, \
    TokenAndPositionEmbedding
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer

logger = logging.getLogger(__name__)


ENCODER_REGISTRY = Registry()


class SequenceEncoder(Encoder, ABC):
    @classmethod
    def register(cls, name):
        ENCODER_REGISTRY[name] = cls


@register_default(name='passthrough')
class SequencePassthroughEncoder(SequenceEncoder):

    def __init__(
            self,
            reduce_output=None,
            **kwargs
    ):
        """
            :param reduce_output: defines how to reduce the output tensor along
                   the `s` sequence length dimention if the rank of the tensor
                   is greater than 2. Available values are: `sum`,
                   `mean` or `avg`, `max`, `concat` (concatenates along
                   the first dimension), `last` (returns the last vector of the
                   first dimension) and `None` or `null` (which does not reduce
                   and returns the full tensor).
            :type reduce_output: str
        """
        super(SequencePassthroughEncoder, self).__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True

    def call(
            self,
            input_sequence,
            training=True,
            mask=None
    ):
        """
            :param input_sequence: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type input_sequence: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        input_sequence = tf.cast(input_sequence, tf.float32)
        while len(input_sequence.shape) < 3:
            input_sequence = tf.expand_dims(
                input_sequence, -1
            )
        hidden = self.reduce_sequence(input_sequence)

        return {'encoder_output': hidden}


@register(name='embed')
class SequenceEmbedEncoder(SequenceEncoder):

    def __init__(
            self,
            vocab,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            weights_initializer=None,
            weights_regularizer=None,
            dropout=0,
            reduce_output='sum',
            **kwargs
    ):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
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
            :param weights_initializer: the initializer to use. If `None`, the default
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
            :type weights_initializer: str
            :param regularize: if `True` the embedding wieghts are added to
                   the set of weights that get reularized by a regularization
                   loss (if the `regularization_lambda` in `training`
                   is greater than 0).
            :type regularize: Boolean
            :param reduce_output: defines how to reduce the output tensor along
                   the `s` sequence length dimention if the rank of the tensor
                   is greater than 2. Available values are: `sum`,
                   `mean` or `avg`, `max`, `concat` (concatenates along
                   the first dimension), `last` (returns the last vector of the
                   first dimension) and `None` or `null` (which does not reduce
                   and returns the full tensor).
            :type reduce_output: str
            :param weights_regularizer: The regularizer to use for the weights
                   of the encoder.
            :type weights_regularizer:
            :param dropout: Tensor (tf.float) of the probability of dropout
            :type dropout: Tensor

        """
        super(SequenceEmbedEncoder, self).__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = reduce_output
        if self.reduce_output is None:
            self.supports_masking = True

        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)

        logger.debug('  EmbedSequence')
        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
            embedding_regularizer=weights_regularizer
        )

    def call(self, inputs, training=None, mask=None):
        """
            :param inputs: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type inputs: Tensor
            :param training: specifying if in training mode
                   (important for dropout)
            :type training: Boolean
        """
        # ================ Embeddings ================
        embedded_sequence = self.embed_sequence(
            inputs, training=training, mask=mask
        )

        hidden = self.reduce_sequence(embedded_sequence)

        return {'encoder_output': hidden}


@register(name='parallel_cnn')
class ParallelCNN(SequenceEncoder):

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
            pool_function='max',
            pool_size=None,
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
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
            reduce_output='max',
            **kwargs):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param conv_layers: it is a list of dictionaries containing
                   the parameters of all the convolutional layers. The length
                   of the list determines the number of parallel convolutional
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `filter_size`, `num_filters`, `pool`,
                   `norm`, `activation` and `regularize`. If any of those values
                   is missing from the dictionary, the default one specified
                   as a parameter of the encoder will be used instead. If both
                   `conv_layers` and `num_conv_layers` are `None`, a default
                   list will be assigned to `conv_layers` with the value
                   `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4},
                   {filter_size: 5}]`.
            :type conv_layers: List
            :param num_conv_layers: if `conv_layers` is `None`, this is
                   the number of parallel convolutional layers.
            :type num_conv_layers: Integer
            :param filter_size:  if a `filter_size` is not already specified in
                   `conv_layers` this is the default `filter_size` that
                   will be used for each layer. It indicates how wide is
                   the 1d convolutional filter.
            :type filter_size: Integer
            :param num_filters: if a `num_filters` is not already specified in
                   `conv_layers` this is the default `num_filters` that
                   will be used for each layer. It indicates the number
                   of filters, and by consequence the output channels of
                   the 1d convolution.
            :type num_filters: Integer
            :param pool_size: if a `pool_size` is not already specified
                  in `conv_layers` this is the default `pool_size` that
                  will be used for each layer. It indicates the size of
                  the max pooling that will be performed along the `s` sequence
                  dimension after the convolution operation.
            :type pool_size: Integer
            :param fc_layers: it is a list of dictionaries containing
                   the parameters of all the fully connected layers. The length
                   of the list determines the number of stacked fully connected
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `fc_size`, `norm`, `activation` and
                   `regularize`. If any of those values is missing from
                   the dictionary, the default one specified as a parameter of
                   the encoder will be used instead. If both `fc_layers` and
                   `num_fc_layers` are `None`, a default list will be assigned
                   to `fc_layers` with the value
                   `[{fc_size: 512}, {fc_size: 256}]`
                   (only applies if `reduce_output` is not `None`).
            :type fc_layers: List
            :param num_fc_layers: if `fc_layers` is `None`, this is the number
                   of stacked fully connected layers (only applies if
                   `reduce_output` is not `None`).
            :type num_fc_layers: Integer
            :param fc_size: if a `fc_size` is not already specified in
                   `fc_layers` this is the default `fc_size` that will be used
                   for each layer. It indicates the size of the output
                   of a fully connected layer.
            :type fc_size: Integer
            :param norm: if a `norm` is not already specified in `conv_layers`
                   or `fc_layers` this is the default `norm` that will be used
                   for each layer. It indicates the norm of the output.
            :type norm: str
            :param activation: Default activation function to use
            :type activation: Str
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(ParallelCNN, self).__init__()
        logger.debug(' {}'.format(self.name))

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
                'Invalid layer parametrization, use either conv_layers or'
                ' num_conv_layers'
            )

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [
                {'fc_size': 512},
                {'fc_size': 256}
            ]
            num_fc_layers = 2
        elif fc_layers is not None and num_fc_layers is not None:
            raise ValueError(
                'Invalid layer parametrization, use either fc_layers or '
                'num_fc_layers only. Not both.'
            )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug('  EmbedSequence')
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
                embedding_regularizer=weights_regularizer
            )

        logger.debug('  ParallelConv1D')
        self.parallel_conv1d = ParallelConv1D(
            layers=self.conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=None,
            # default_bias_constraint=None,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_padding='same',
        )

        if self.reduce_output is not None:
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
            :param inputs: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int
            :type inputs: Tensor
            :param training: bool specifying if in training mode (important for dropout)
            :type training: bool
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(
                inputs, training=training, mask=mask
            )
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = tf.expand_dims(embedded_sequence, -1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.parallel_conv1d(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return {'encoder_output': hidden}


@register(name='stacked_cnn')
class StackedCNN(SequenceEncoder):

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
            num_filters=256,
            filter_size=5,
            strides=1,
            padding='same',
            dilation_rate=1,
            pool_function='max',
            pool_size=None,
            pool_strides=None,
            pool_padding='same',
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
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
            reduce_output='max',
            **kwargs
    ):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param conv_layers: it is a list of dictionaries containing
                   the parameters of all the convolutional layers. The length
                   of the list determines the number of parallel convolutional
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `filter_size`, `num_filters`, `pool`,
                   `norm`, `activation` and `regularize`. If any of those values
                   is missing from the dictionary, the default one specified
                   as a parameter of the encoder will be used instead. If both
                   `conv_layers` and `num_conv_layers` are `None`, a default
                   list will be assigned to `conv_layers` with the value
                   `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4},
                   {filter_size: 5}]`.
            :type conv_layers: List
            :param num_conv_layers: if `conv_layers` is `None`, this is
                   the number of stacked convolutional layers.
            :type num_conv_layers: Integer
            :param filter_size:  if a `filter_size` is not already specified in
                   `conv_layers` this is the default `filter_size` that
                   will be used for each layer. It indicates how wide is
                   the 1d convolutional filter.
            :type filter_size: Integer
            :param num_filters: if a `num_filters` is not already specified in
                   `conv_layers` this is the default `num_filters` that
                   will be used for each layer. It indicates the number
                   of filters, and by consequence the output channels of
                   the 1d convolution.
            :type num_filters: Integer
            :param pool_size: if a `pool_size` is not already specified
                  in `conv_layers` this is the default `pool_size` that
                  will be used for each layer. It indicates the size of
                  the max pooling that will be performed along the `s` sequence
                  dimension after the convolution operation.
            :type pool_size: Integer
            :param fc_layers: it is a list of dictionaries containing
                   the parameters of all the fully connected layers. The length
                   of the list determines the number of stacked fully connected
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `fc_size`, `norm`, `activation` and
                   `regularize`. If any of those values is missing from
                   the dictionary, the default one specified as a parameter of
                   the encoder will be used instead. If both `fc_layers` and
                   `num_fc_layers` are `None`, a default list will be assigned
                   to `fc_layers` with the value
                   `[{fc_size: 512}, {fc_size: 256}]`
                   (only applies if `reduce_output` is not `None`).
            :type fc_layers: List
            :param num_fc_layers: if `fc_layers` is `None`, this is the number
                   of stacked fully connected layers (only applies if
                   `reduce_output` is not `None`).
            :type num_fc_layers: Integer
            :param fc_size: if a `fc_size` is not already specified in
                   `fc_layers` this is the default `fc_size` that will be used
                   for each layer. It indicates the size of the output
                   of a fully connected layer.
            :type fc_size: Integer
            :param norm: if a `norm` is not already specified in `conv_layers`
                   or `fc_layers` this is the default `norm` that will be used
                   for each layer. It indicates the norm of the output.
            :type norm: str
            :param activation: Default activation function to use
            :type activation: Str
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(StackedCNN, self).__init__()
        logger.debug(' {}'.format(self.name))

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
                'Invalid layer parametrization, use either conv_layers or '
                'num_conv_layers'
            )

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [
                {'fc_size': 512},
                {'fc_size': 256}
            ]
            num_fc_layers = 2
        elif fc_layers is not None and num_fc_layers is not None:
            raise ValueError(
                'Invalid layer parametrization, use either fc_layers or '
                'num_fc_layers only. Not both.'
            )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug('  EmbedSequence')
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
                embedding_regularizer=weights_regularizer
            )

        logger.debug('  Conv1DStack')
        self.conv1d_stack = Conv1DStack(
            layers=self.conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_strides=strides,
            default_padding=padding,
            default_dilation_rate=dilation_rate,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=None,
            # default_bias_constraint=None,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_strides=pool_strides,
            default_pool_padding=pool_padding,
        )

        if self.reduce_output is not None:
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
            :param input_sequence: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type input_sequence: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout: Tensor (tf.float) of the probability of dropout
            :type dropout: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(
                inputs, training=training, mask=mask
            )
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = tf.expand_dims(embedded_sequence, -1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.conv1d_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return {'encoder_output': hidden}


@register(name='stacked_parallel_cnn')
class StackedParallelCNN(SequenceEncoder):

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
            pool_function='max',
            pool_size=None,
            fc_layers=None,
            num_fc_layers=None,
            fc_size=256,
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
            reduce_output='max',
            **kwargs
    ):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param stacked_layers: it is a of lists of list of dictionaries
                   containing the parameters of the stack of
                   parallel convolutional layers. The length of the list
                   determines the number of stacked parallel
                   convolutional layers, length of the sub-lists determines
                   the number of parallel conv layers and the content
                   of each dictionary determines the parameters for
                   a specific layer. The available parameters for each layer are:
                   `filter_size`, `num_filters`, `pool_size`, `norm`,
                   `activation` and `regularize`. If any of those values
                   is missing from the dictionary, the default one specified
                   as a parameter of the encoder will be used instead. If both
                   `stacked_layers` and `num_stacked_layers` are `None`,
                   a default list will be assigned to `stacked_layers` with
                   the value `[[{filter_size: 2}, {filter_size: 3},
                   {filter_size: 4}, {filter_size: 5}], [{filter_size: 2},
                   {filter_size: 3}, {filter_size: 4}, {filter_size: 5}],
                   [{filter_size: 2}, {filter_size: 3}, {filter_size: 4},
                   {filter_size: 5}]]`.
            :type stacked_layers: List
            :param num_stacked_layers: if `stacked_layers` is `None`, this is
                   the number of elements in the stack of
                   parallel convolutional layers.
            :type num_stacked_layers: Integer
            :param filter_size:  if a `filter_size` is not already specified in
                   `conv_layers` this is the default `filter_size` that
                   will be used for each layer. It indicates how wide is
                   the 1d convolutional filter.
            :type filter_size: Integer
            :param num_filters: if a `num_filters` is not already specified in
                   `conv_layers` this is the default `num_filters` that
                   will be used for each layer. It indicates the number
                   of filters, and by consequence the output channels of
                   the 1d convolution.
            :type num_filters: Integer
            :param pool_size: if a `pool_size` is not already specified
                  in `conv_layers` this is the default `pool_size` that
                  will be used for each layer. It indicates the size of
                  the max pooling that will be performed along the `s` sequence
                  dimension after the convolution operation.
            :type pool_size: Integer
            :param fc_layers: it is a list of dictionaries containing
                   the parameters of all the fully connected layers. The length
                   of the list determines the number of stacked fully connected
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `fc_size`, `norm`, `activation` and
                   `regularize`. If any of those values is missing from
                   the dictionary, the default one specified as a parameter of
                   the encoder will be used instead. If both `fc_layers` and
                   `num_fc_layers` are `None`, a default list will be assigned
                   to `fc_layers` with the value
                   `[{fc_size: 512}, {fc_size: 256}]`
                   (only applies if `reduce_output` is not `None`).
            :type fc_layers: List
            :param num_fc_layers: if `fc_layers` is `None`, this is the number
                   of stacked fully connected layers (only applies if
                   `reduce_output` is not `None`).
            :type num_fc_layers: Integer
            :param fc_size: if a `fc_size` is not already specified in
                   `fc_layers` this is the default `fc_size` that will be used
                   for each layer. It indicates the size of the output
                   of a fully connected layer.
            :type fc_size: Integer
            :param norm: if a `norm` is not already specified in `conv_layers`
                   or `fc_layers` this is the default `norm` that will be used
                   for each layer. It indicates the norm of the output.
            :type norm: str
            :param activation: Default activation function to use
            :type activation: Str
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(StackedParallelCNN, self).__init__()
        logger.debug(' {}'.format(self.name))

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
                'Invalid layer parametrization, use either stacked_layers or'
                ' num_stacked_layers'
            )

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [
                {'fc_size': 512},
                {'fc_size': 256}
            ]
            num_fc_layers = 2
        elif fc_layers is not None and num_fc_layers is not None:
            raise ValueError(
                'Invalid layer parametrization, use either fc_layers or '
                'num_fc_layers only. Not both.'
            )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug('  EmbedSequence')
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
                embedding_regularizer=weights_regularizer
            )

        logger.debug('  ParallelConv1DStack')
        self.parallel_conv1d_stack = ParallelConv1DStack(
            stacked_layers=self.stacked_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
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
            default_pool_function=pool_function,
            default_pool_size=pool_size,
        )

        if self.reduce_output is not None:
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
            :param inputs: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type inputs: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout: Tensor (tf.float) of the probability of dropout
            :type dropout: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(
                inputs, training=training, mask=mask
            )
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = tf.expand_dims(embedded_sequence, -1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.parallel_conv1d_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return {'encoder_output': hidden}


@register(name='rnn')
class StackedRNN(SequenceEncoder):

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
            activation='tanh',
            recurrent_activation='sigmoid',
            unit_forget_bias=True,
            recurrent_initializer='orthogonal',
            recurrent_regularizer=None,
            # recurrent_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=256,
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
            fc_activation='relu',
            fc_dropout=0,
            reduce_output='last',
            **kwargs
    ):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param conv_layers: it is a list of dictionaries containing
                   the parameters of all the convolutional layers. The length
                   of the list determines the number of parallel convolutional
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `filter_size`, `num_filters`, `pool`,
                   `norm`, `activation` and `regularize`. If any of those values
                   is missing from the dictionary, the default one specified
                   as a parameter of the encoder will be used instead. If both
                   `conv_layers` and `num_conv_layers` are `None`, a default
                   list will be assigned to `conv_layers` with the value
                   `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4},
                   {filter_size: 5}]`.
            :type conv_layers: List
            :param num_conv_layers: if `conv_layers` is `None`, this is
                   the number of stacked convolutional layers.
            :type num_conv_layers: Integer
            :param filter_size:  if a `filter_size` is not already specified in
                   `conv_layers` this is the default `filter_size` that
                   will be used for each layer. It indicates how wide is
                   the 1d convolutional filter.
            :type filter_size: Integer
            :param num_filters: if a `num_filters` is not already specified in
                   `conv_layers` this is the default `num_filters` that
                   will be used for each layer. It indicates the number
                   of filters, and by consequence the output channels of
                   the 1d convolution.
            :type num_filters: Integer
            :param pool_size: if a `pool_size` is not already specified
                  in `conv_layers` this is the default `pool_size` that
                  will be used for each layer. It indicates the size of
                  the max pooling that will be performed along the `s` sequence
                  dimension after the convolution operation.
            :type pool_size: Integer
            :param num_rec_layers: the number of stacked recurrent layers.
            :type num_rec_layers: Integer
            :param cell_type: the type of recurrent cell to use.
                   Avalable values are: `rnn`, `lstm`, `lstm_block`, `lstm`,
                   `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`.
                   For reference about the differences between the cells please
                   refer to TensorFlow's documentstion. We suggest to use the
                   `block` variants on CPU and the `cudnn` variants on GPU
                   because of their increased speed.
            :type cell_type: str
            :param state_size: the size of the state of the rnn.
            :type state_size: Integer
            :param bidirectional: if `True` two recurrent networks will perform
                   encoding in the forward and backward direction and
                   their outputs will be concatenated.
            :type bidirectional: Boolean
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(StackedRNN, self).__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True

        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug('  EmbedSequence')
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=fc_dropout,
                embedding_initializer=weights_initializer,
                embedding_regularizer=weights_regularizer
            )

        logger.debug('  RecurrentStack')
        self.recurrent_stack = RecurrentStack(
            state_size=state_size,
            cell_type=cell_type,
            num_layers=num_layers,
            bidirectional=bidirectional,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
            weights_initializer=weights_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            # kernel_constraint=kernel_constraint,
            # recurrent_constraint=recurrent_constraint,
            # bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

        if self.reduce_output is not None:
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
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    def call(self, inputs, training=None, mask=None):
        """
            :param input_sequence: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type input_sequence: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout: Tensor (tf.float) of the probability of dropout
            :type dropout: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(
                inputs, training=training, mask=mask
            )
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = tf.expand_dims(embedded_sequence, -1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Recurrent Layers ================
        hidden, final_state = self.recurrent_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return {
            'encoder_output': hidden,
            'encoder_output_state': final_state
        }


@register(name='cnnrnn')
class StackedCNNRNN(SequenceEncoder):

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
            num_conv_layers=1,
            num_filters=256,
            filter_size=5,
            strides=1,
            padding='same',
            dilation_rate=1,
            conv_activation='relu',
            conv_dropout=0.0,
            pool_function='max',
            pool_size=2,
            pool_strides=None,
            pool_padding='same',
            num_rec_layers=1,
            state_size=256,
            cell_type='rnn',
            bidirectional=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            unit_forget_bias=True,
            recurrent_initializer='orthogonal',
            recurrent_regularizer=None,
            # recurrent_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=256,
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
            fc_activation='relu',
            fc_dropout=0,
            reduce_output='last',
            **kwargs
    ):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param num_layers: the number of stacked recurrent layers.
            :type num_layers: Integer
            :param cell_type: the type of recurrent cell to use.
                   Avalable values are: `rnn`, `lstm`, `lstm_block`, `lstm`,
                   `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`.
                   For reference about the differences between the cells please
                   refer to TensorFlow's documentstion. We suggest to use the
                   `block` variants on CPU and the `cudnn` variants on GPU
                   because of their increased speed.
            :type cell_type: str
            :param state_size: the size of the state of the rnn.
            :type state_size: Integer
            :param bidirectional: if `True` two recurrent networks will perform
                   encoding in the forward and backward direction and
                   their outputs will be concatenated.
            :type bidirectional: Boolean
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(StackedCNNRNN, self).__init__()
        logger.debug(' {}'.format(self.name))

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
                'Invalid layer parametrization, use either conv_layers or '
                'num_conv_layers'
            )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug('  EmbedSequence')
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=fc_dropout,
                embedding_initializer=weights_initializer,
                embedding_regularizer=weights_regularizer
            )

        logger.debug('  Conv1DStack')
        self.conv1d_stack = Conv1DStack(
            layers=self.conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_strides=strides,
            default_padding=padding,
            default_dilation_rate=dilation_rate,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_weights_regularizer=weights_regularizer,
            default_bias_regularizer=bias_regularizer,
            default_activity_regularizer=activity_regularizer,
            # default_weights_constraint=None,
            # default_bias_constraint=None,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=conv_activation,
            default_dropout=conv_dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_strides=pool_strides,
            default_pool_padding=pool_padding,
        )

        logger.debug('  RecurrentStack')
        self.recurrent_stack = RecurrentStack(
            state_size=state_size,
            cell_type=cell_type,
            num_layers=num_rec_layers,
            bidirectional=bidirectional,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
            weights_initializer=weights_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            # kernel_constraint=kernel_constraint,
            # recurrent_constraint=recurrent_constraint,
            # bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

        if self.reduce_output is not None:
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
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    def call(self, inputs, training=None, mask=None):
        """
            :param input_sequence: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type input_sequence: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout: Tensor (tf.float) of the probability of dropout
            :type dropout: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(
                inputs, training=training, mask=mask
            )
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = tf.expand_dims(embedded_sequence, -1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.conv1d_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Recurrent Layers ================
        hidden, final_state = self.recurrent_stack(
            hidden,
            training=training
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return {
            'encoder_output': hidden,
            'encoder_output_state': final_state
        }


@register(name='transformer')
class StackedTransformer(SequenceEncoder):

    def __init__(
            self,
            max_sequence_length,
            should_embed=True,
            vocab=None,
            representation='dense',
            embedding_size=256,
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=False,
            num_layers=1,
            hidden_size=256,
            num_heads=8,
            transformer_fc_size=256,
            dropout=0.1,
            fc_layers=None,
            num_fc_layers=0,
            fc_size=256,
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
            fc_activation='relu',
            fc_dropout=0,
            reduce_output='last',
            **kwargs
    ):
        """
            :param should_embed: If True the input sequence is expected
                   to be made of integers and will be mapped into embeddings
            :type should_embed: Boolean
            :param vocab: Vocabulary of the input feature to encode
            :type vocab: List
            :param representation: the possible values are `dense` and `sparse`.
                   `dense` means the mebeddings are initialized randomly,
                   `sparse` meanse they are initialized to be one-hot encodings.
            :type representation: Str (one of 'dense' or 'sparse')
            :param embedding_size: it is the maximum embedding size, the actual
                   size will be `min(vocaularyb_size, embedding_size)`
                   for `dense` representations and exacly `vocaularyb_size`
                   for the `sparse` encoding, where `vocabulary_size` is
                   the number of different strings appearing in the training set
                   in the column the feature is named after (plus 1 for `<UNK>`).
            :type embedding_size: Integer
            :param embeddings_trainable: If `True` embeddings are trained during
                   the training process, if `False` embeddings are fixed.
                   It may be useful when loading pretrained embeddings
                   for avoiding finetuning them. This parameter has effect only
                   for `representation` is `dense` as `sparse` one-hot encodings
                    are not trainable.
            :type embeddings_trainable: Boolean
            :param pretrained_embeddings: by default `dense` embeddings
                   are initialized randomly, but this parameter allows to specify
                   a path to a file containing embeddings in the GloVe format.
                   When the file containing the embeddings is loaded, only the
                   embeddings with labels present in the vocabulary are kept,
                   the others are discarded. If the vocabulary contains strings
                   that have no match in the embeddings file, their embeddings
                   are initialized with the average of all other embedding plus
                   some random noise to make them different from each other.
                   This parameter has effect only if `representation` is `dense`.
            :type pretrained_embeddings: str (filepath)
            :param embeddings_on_cpu: by default embedings matrices are stored
                   on GPU memory if a GPU is used, as it allows
                   for faster access, but in some cases the embedding matrix
                   may be really big and this parameter forces the placement
                   of the embedding matrix in regular memroy and the CPU is used
                   to resolve them, slightly slowing down the process
                   as a result of data transfer between CPU and GPU memory.
            :param conv_layers: it is a list of dictionaries containing
                   the parameters of all the convolutional layers. The length
                   of the list determines the number of parallel convolutional
                   layers and the content of each dictionary determines
                   the parameters for a specific layer. The available parameters
                   for each layer are: `filter_size`, `num_filters`, `pool`,
                   `norm`, `activation` and `regularize`. If any of those values
                   is missing from the dictionary, the default one specified
                   as a parameter of the encoder will be used instead. If both
                   `conv_layers` and `num_conv_layers` are `None`, a default
                   list will be assigned to `conv_layers` with the value
                   `[{filter_size: 2}, {filter_size: 3}, {filter_size: 4},
                   {filter_size: 5}]`.
            :type conv_layers: List
            :param num_conv_layers: if `conv_layers` is `None`, this is
                   the number of stacked convolutional layers.
            :type num_conv_layers: Integer
            :param filter_size:  if a `filter_size` is not already specified in
                   `conv_layers` this is the default `filter_size` that
                   will be used for each layer. It indicates how wide is
                   the 1d convolutional filter.
            :type filter_size: Integer
            :param num_filters: if a `num_filters` is not already specified in
                   `conv_layers` this is the default `num_filters` that
                   will be used for each layer. It indicates the number
                   of filters, and by consequence the output channels of
                   the 1d convolution.
            :type num_filters: Integer
            :param pool_size: if a `pool_size` is not already specified
                  in `conv_layers` this is the default `pool_size` that
                  will be used for each layer. It indicates the size of
                  the max pooling that will be performed along the `s` sequence
                  dimension after the convolution operation.
            :type pool_size: Integer
            :param num_rec_layers: the number of stacked recurrent layers.
            :type num_rec_layers: Integer
            :param cell_type: the type of recurrent cell to use.
                   Avalable values are: `rnn`, `lstm`, `lstm_block`, `lstm`,
                   `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`.
                   For reference about the differences between the cells please
                   refer to TensorFlow's documentstion. We suggest to use the
                   `block` variants on CPU and the `cudnn` variants on GPU
                   because of their increased speed.
            :type cell_type: str
            :param state_size: the size of the state of the rnn.
            :type state_size: Integer
            :param bidirectional: if `True` two recurrent networks will perform
                   encoding in the forward and backward direction and
                   their outputs will be concatenated.
            :type bidirectional: Boolean
            :param dropout: determines if there should be a dropout layer before
                   returning the encoder output.
            :type dropout: Boolean
            :param initializer: the initializer to use. If `None` it uses
                   `glorot_uniform`. Options are: `constant`, `identity`,
                   `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
                   `truncated_normal`, `variance_scaling`, `glorot_normal`,
                   `glorot_uniform`, `xavier_normal`, `xavier_uniform`,
                   `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                   Alternatively it is possible to specify a dictionary with
                   a key `type` that identifies the type of initialzier and
                   other keys for its parameters,
                   e.g. `{type: normal, mean: 0, stddev: 0}`.
                   To know the parameters of each initializer, please refer
                   to TensorFlow's documentation.
            :type initializer: str
            :param regularize: if a `regularize` is not already specified in
                   `conv_layers` or `fc_layers` this is the default `regularize`
                   that will be used for each layer. It indicates if
                   the layer weights should be considered when computing
                   a regularization loss.
            :type regularize:
            :param reduce_output: defines how to reduce the output tensor of
                   the convolutional layers along the `s` sequence length
                   dimention if the rank of the tensor is greater than 2.
                   Available values are: `sum`, `mean` or `avg`, `max`, `concat`
                   (concatenates along the first dimension), `last` (returns
                   the last vector of the first dimension) and `None` or `null`
                   (which does not reduce and returns the full tensor).
            :type reduce_output: str
        """
        super(StackedTransformer, self).__init__()
        logger.debug(' {}'.format(self.name))

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_output)
        if self.reduce_output is None:
            self.supports_masking = True

        self.should_embed = should_embed
        self.should_project = False
        self.embed_sequence = None

        if self.should_embed:
            logger.debug('  EmbedSequence')
            self.embed_sequence = TokenAndPositionEmbedding(
                max_sequence_length,
                vocab,
                embedding_size,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
                embedding_regularizer=weights_regularizer
            )

            if embedding_size != hidden_size:
                logger.debug('  project_to_embed_size Dense')
                self.project_to_hidden_size = Dense(hidden_size)
                self.should_project = True
        else:
            logger.debug('  project_to_embed_size Dense')
            self.project_to_hidden_size = Dense(hidden_size)
            self.should_project = True

        logger.debug('  TransformerStack')
        self.transformer_stack = TrasformerStack(
            hidden_size=hidden_size,
            num_heads=num_heads,
            fc_size=transformer_fc_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if self.reduce_output is not None:
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
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    def call(self, inputs, training=None, mask=None):
        """
            :param input_sequence: The input sequence fed into the encoder.
                   Shape: [batch x sequence length], type tf.int32
            :type input_sequence: Tensor
            :param regularizer: The regularizer to use for the weights
                   of the encoder.
            :type regularizer:
            :param dropout: Tensor (tf.float) of the probability of dropout
            :type dropout: Tensor
            :param is_training: Tesnor (tf.bool) specifying if in training mode
                   (important for dropout)
            :type is_training: Tensor
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(
                inputs, training=training, mask=mask
            )
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = tf.expand_dims(embedded_sequence, -1)

        # shape=(?, sequence_length, embedding_size)
        if self.should_project:
            hidden = self.project_to_hidden_size(embedded_sequence)
        else:
            hidden = embedded_sequence
        # shape=(?, sequence_length, hidden)

        # ================ Transformer Layers ================
        hidden = self.transformer_stack(
            hidden,
            training=training,
            mask=mask
        )

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(
                hidden,
                training=training,
                mask=mask
            )

        return {'encoder_output': hidden}
