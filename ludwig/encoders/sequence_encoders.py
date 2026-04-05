#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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

import torch
from torch import nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, ENCODER_OUTPUT, ENCODER_OUTPUT_STATE, SEQUENCE, TEXT, TIMESERIES
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder, register_sequence_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.modules.attention_modules import TransformerStack
from ludwig.modules.convolutional_modules import Conv1DStack, ParallelConv1D, ParallelConv1DStack
from ludwig.modules.embedding_modules import EmbedSequence, TokenAndPositionEmbedding
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.encoders.sequence_encoders import (
    MambaEncoderConfig,
    ParallelCNNConfig,
    SequenceEmbedConfig,
    SequenceEncoderConfig,
    SequencePassthroughConfig,
    StackedCNNConfig,
    StackedCNNRNNConfig,
    StackedParallelCNNConfig,
    StackedRNNConfig,
    StackedTransformerConfig,
)

logger = logging.getLogger(__name__)


class SequenceEncoder(Encoder):
    """Base class for all sequence encoders in Ludwig.

    Sequence encoders take a sequence of tokens (integers) or a sequence of vectors (floats) and produce a fixed-size or
    variable-length hidden representation.
    """

    pass


@DeveloperAPI
@register_encoder("passthrough", [SEQUENCE, TEXT, TIMESERIES])
class SequencePassthroughEncoder(SequenceEncoder):
    """Identity encoder that passes through the input sequence without transformation.

    The input sequence is cast to float32 and optionally reduced along the sequence
    dimension. This is useful when the input features are already pre-computed embeddings
    or continuous-valued sequences (e.g., timeseries) that do not need learned encoding.

    Use when: input features are already meaningful representations (pre-computed embeddings,
    raw timeseries values) and no learned transformation is needed.

    Alternatives: SequenceEmbedEncoder (if inputs are token IDs needing embeddings),
    any other sequence encoder (if learned feature extraction is desired).

    Performance: O(1) computation (no learnable parameters beyond optional reduction).
    """

    def __init__(
        self,
        reduce_output: str = None,
        max_sequence_length: int = 256,
        encoding_size: int = None,
        encoder_config=None,
        **kwargs,
    ):
        """
        :param reduce_output: defines how to reduce the output tensor along
               the `s` sequence length dimension if the rank of the tensor
               is greater than 2. Available values are: `sum`,
               `mean` or `avg`, `max`, `concat` (concatenates along
               the first dimension), `last` (returns the last vector of the
               first dimension) and `None` or `null` (which does not reduce
               and returns the full tensor).
        :param max_sequence_length: The maximum sequence length.
        :param encoding_size: The size of the encoding vector, or None if sequence elements are scalars.
        """
        super().__init__()
        self.config = encoder_config
        self.max_sequence_length = max_sequence_length

        logger.debug(f" {self.name}")

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output, max_sequence_length=max_sequence_length, encoding_size=encoding_size
        )
        if self.reduce_output is None:
            self.supports_masking = True

    def forward(self, input_sequence: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param input_sequence: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32 or
                      [batch x sequence length x encoding size], type torch.float32
        :type input_sequence: Tensor
        :param mask: Sequence mask (not yet implemented).
               Shape: [batch x sequence length]
        :type mask: Tensor
        """
        input_sequence = input_sequence.type(torch.float32)
        while len(input_sequence.shape) < 3:
            input_sequence = input_sequence.unsqueeze(-1)
        hidden = self.reduce_sequence(input_sequence)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return SequencePassthroughConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape


@DeveloperAPI
@register_encoder("embed", [SEQUENCE, TEXT])
class SequenceEmbedEncoder(SequenceEncoder):
    """Embedding-only encoder that maps token sequences to embeddings and reduces them.

    Maps each token in the input sequence to a dense embedding vector, then reduces the
    sequence of embeddings to a single vector using the specified reduce_output strategy
    (default: sum). No convolutional, recurrent, or attention layers are applied.

    Use when: a simple bag-of-words style representation is sufficient. Works well for
    short sequences or when the order of tokens is not important. Very fast and memory-efficient.

    Alternatives: ParallelCNN or StackedCNN (to capture local n-gram patterns),
    StackedRNN (to capture sequential dependencies), StackedTransformer (to capture
    long-range dependencies with attention).

    Performance: O(n) time and memory where n is sequence length. No sequential dependencies
    computed, making it the fastest learned sequence encoder.
    """

    def __init__(
        self,
        vocab,
        max_sequence_length,
        representation="dense",
        embedding_size=256,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        weights_initializer=None,
        dropout=0,
        reduce_output="sum",
        encoder_config=None,
        **kwargs,
    ):
        """
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param max_sequence_length: The maximum sequence length.
        :type max_sequence_length: int
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
               on GPU memory if a GPU is used, as it allows
               for faster access, but in some cases the embedding matrix
               may be really big and this parameter forces the placement
               of the embedding matrix in regular memory and the CPU is used
               to resolve them, slightly slowing down the process
               as a result of data transfer between CPU and GPU memory.
        :type embeddings_on_cpu: Boolean
        :param weights_initializer: the initializer to use. If `None`, the default
               initialized of each variable is used (`xavier_uniform`
               in most cases). Options are: `constant`, `identity`, `zeros`,
                `ones`, `orthogonal`, `normal`, `uniform`,
                `truncated_normal`, `variance_scaling`, `xavier_normal`,
                `xavier_uniform`, `xavier_normal`,
                `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
                Alternatively it is possible to specify a dictionary with
                a key `type` that identifies the type of initializer and
                other keys for its parameters, e.g.
                `{type: normal, mean: 0, stddev: 0}`.
                To know the parameters of each initializer, please refer to
                PyTorch's documentation.
        :type weights_initializer: str
        :param dropout: Tensor (torch.float) The dropout probability.
        :type dropout: Tensor
        :param reduce_output: defines how to reduce the output tensor along
               the `s` sequence length dimension if the rank of the tensor
               is greater than 2. Available values are: `sum`,
               `mean` or `avg`, `max`, `concat` (concatenates along
               the first dimension), `last` (returns the last vector of the
               first dimension) and `None` or `null` (which does not reduce
               and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length

        self.reduce_output = reduce_output
        if self.reduce_output is None:
            self.supports_masking = True

        logger.debug("  EmbedSequence")
        self.embed_sequence = EmbedSequence(
            vocab,
            embedding_size,
            max_sequence_length=max_sequence_length,
            representation=representation,
            embeddings_trainable=embeddings_trainable,
            pretrained_embeddings=pretrained_embeddings,
            embeddings_on_cpu=embeddings_on_cpu,
            dropout=dropout,
            embedding_initializer=weights_initializer,
        )

        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=max_sequence_length,
            encoding_size=self.embed_sequence.output_shape[-1],
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented in EmbedSequence)
        """
        embedded_sequence = self.embed_sequence(inputs, mask=mask)
        hidden = self.reduce_sequence(embedded_sequence)
        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return SequenceEmbedConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return self.reduce_sequence.output_shape


@DeveloperAPI
@register_sequence_encoder("parallel_cnn")
@register_encoder("parallel_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class ParallelCNN(SequenceEncoder):
    """Parallel 1D CNN encoder with multiple filter sizes applied simultaneously.

    Embeds the input sequence, then passes it through multiple parallel 1D convolutional
    layers with different filter sizes (default: 2, 3, 4, 5). The outputs are pooled and
    concatenated, then passed through fully connected layers. This architecture captures
    n-gram patterns of different lengths simultaneously, similar to the approach in
    "Convolutional Neural Networks for Sentence Classification" (Kim, 2014).

    Reference: Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
    https://arxiv.org/abs/1408.5882

    Use when: capturing local n-gram patterns of varying lengths is important (e.g., text
    classification, sentiment analysis). Good default choice for sequence/text features.

    Alternatives: StackedCNN (deeper sequential convolutions for hierarchical features),
    StackedRNN (when sequential order matters more than local patterns),
    StackedTransformer (when long-range dependencies are important).

    Performance: O(n * k * f) where n is sequence length, k is max filter size, f is
    num_filters. Parallelizable across filters. Memory-efficient compared to RNNs/Transformers.
    """

    def __init__(
        self,
        should_embed=True,
        vocab=None,
        representation="dense",
        embedding_size=256,
        max_sequence_length=None,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        conv_layers=None,
        num_conv_layers=None,
        filter_size=3,
        num_filters=256,
        pool_function="max",
        pool_size=None,
        fc_layers=None,
        num_fc_layers=None,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        activation="relu",
        dropout=0,
        reduce_output="max",
        encoder_config=None,
        **kwargs,
    ):
        # todo: revise docstring
        """
        :param should_embed: If True the input sequence is expected
               to be made of integers and will be mapped into embeddings
        :type should_embed: Boolean
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: Str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
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
               `norm`, and `activation`. If any of those values
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
               for each layer are: `output_size`, `norm` and `activation`.
               If any of those values is missing from
               the dictionary, the default one specified as a parameter of
               the encoder will be used instead. If both `fc_layers` and
               `num_fc_layers` are `None`, a default list will be assigned
               to `fc_layers` with the value
               `[{output_size: 512}, {output_size: 256}]`
               (only applies if `reduce_output` is not `None`).
        :type fc_layers: List
        :param num_fc_layers: if `fc_layers` is `None`, this is the number
               of stacked fully connected layers (only applies if
               `reduce_output` is not `None`).
        :type num_fc_layers: Integer
        :param output_size: if a `output_size` is not already specified in
               `fc_layers` this is the default `output_size` that will be used
               for each layer. It indicates the size of the output
               of a fully connected layer.
        :type output_size: Integer
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
               `xavier_uniform`. Options are: `constant`, `identity`,
               `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
               `truncated_normal`, `variance_scaling`, `xavier_normal`,
               `xavier_uniform`, `xavier_normal`,
               `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
               Alternatively it is possible to specify a dictionary with
               a key `type` that identifies the type of initializer and
               other keys for its parameters,
               e.g. `{type: normal, mean: 0, stddev: 0}`.
               To know the parameters of each initializer, please refer
               to PyTorch's documentation.
        :type initializer: str
        :param reduce_output: defines how to reduce the output tensor of
               the convolutional layers along the `s` sequence length
               dimension if the rank of the tensor is greater than 2.
               Available values are: `sum`, `mean` or `avg`, `max`, `concat`
               (concatenates along the first dimension), `last` (returns
               the last vector of the first dimension) and `None` or `null`
               (which does not reduce and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.max_sequence_length = max_sequence_length

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
            self.conv_layers = [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}]
            self.num_conv_layers = 4
        else:
            raise ValueError("Invalid layer parametrization, use either conv_layers or num_conv_layers")

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [{"output_size": 512}, {"output_size": 256}]
            num_fc_layers = 2
        elif fc_layers is not None and num_fc_layers is not None:
            raise ValueError("Invalid layer parametrization, use either fc_layers or num_fc_layers only. Not both.")

        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                max_sequence_length=max_sequence_length,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )

        logger.debug("  ParallelConv1D")
        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else embedding_size
        self.parallel_conv1d = ParallelConv1D(
            in_channels=in_channels,
            max_sequence_length=self.max_sequence_length,
            layers=self.conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_padding="same",
        )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=max_sequence_length,
            encoding_size=self.parallel_conv1d.output_shape[-1],
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=activation,
                default_dropout=dropout,
            )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)
            embedded_sequence = embedded_sequence.to(dtype=torch.float)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.parallel_conv1d(hidden, mask=mask)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        return {ENCODER_OUTPUT: hidden}

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return ParallelCNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.parallel_conv1d.output_shape


@DeveloperAPI
@register_sequence_encoder("stacked_cnn")
@register_encoder("stacked_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedCNN(SequenceEncoder):
    """Stacked 1D CNN encoder with sequential convolutional layers.

    Embeds the input sequence, then passes it through a stack of 1D convolutional layers
    applied sequentially (each layer's output feeds into the next). Optionally includes
    pooling between layers. The deeper architecture allows learning hierarchical features:
    lower layers capture local patterns while higher layers capture broader patterns.
    Default architecture uses 6 convolutional layers inspired by the VDCNN model.

    Reference: Conneau, A., et al. (2017). Very Deep Convolutional Networks for Text
    Classification. https://arxiv.org/abs/1606.01781

    Use when: hierarchical feature extraction is needed, where lower-level n-gram patterns
    should be composed into higher-level representations. Good for longer sequences where
    progressive pooling can reduce computation.

    Alternatives: ParallelCNN (simpler, captures multiple filter sizes simultaneously),
    StackedParallelCNN (combines both parallel and stacked convolutions),
    StackedRNN (when order/sequential dependencies are crucial).

    Performance: O(n * L * k * f) where n is sequence length, L is num layers, k is filter
    size, f is num_filters. Fully parallelizable. Pooling reduces sequence length progressively.
    """

    def __init__(
        self,
        should_embed=True,
        vocab=None,
        representation="dense",
        embedding_size=256,
        max_sequence_length=None,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        conv_layers=None,
        num_conv_layers=None,
        num_filters=256,
        filter_size=5,
        strides=1,
        # todo: assess how to specify padding for equivalent to 'same'
        padding="same",
        dilation_rate=1,
        pool_function="max",
        pool_size=None,
        pool_strides=None,
        # todo: determine how to pool_padding equivalent of 'same'
        pool_padding="same",
        fc_layers=None,
        num_fc_layers=None,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        activation="relu",
        dropout=0,
        reduce_output="max",
        encoder_config=None,
        **kwargs,
    ):
        # todo: fixup docstring
        """
        :param should_embed: If True the input sequence is expected
               to be made of integers and will be mapped into embeddings
        :type should_embed: Boolean
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: Str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
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
               `norm` and `activation`. If any of those values
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
               for each layer are: `output_size`, `norm` and `activation`.
               If any of those values is missing from
               the dictionary, the default one specified as a parameter of
               the encoder will be used instead. If both `fc_layers` and
               `num_fc_layers` are `None`, a default list will be assigned
               to `fc_layers` with the value
               `[{output_size: 512}, {output_size: 256}]`
               (only applies if `reduce_output` is not `None`).
        :type fc_layers: List
        :param num_fc_layers: if `fc_layers` is `None`, this is the number
               of stacked fully connected layers (only applies if
               `reduce_output` is not `None`).
        :type num_fc_layers: Integer
        :param output_size: if a `output_size` is not already specified in
               `fc_layers` this is the default `output_size` that will be used
               for each layer. It indicates the size of the output
               of a fully connected layer.
        :type output_size: Integer
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
               `xavier_uniform`. Options are: `constant`, `identity`,
               `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
               `truncated_normal`, `variance_scaling`, `xavier_normal`,
               `xavier_uniform`, `xavier_normal`,
               `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
               Alternatively it is possible to specify a dictionary with
               a key `type` that identifies the type of initializer and
               other keys for its parameters,
               e.g. `{type: normal, mean: 0, stddev: 0}`.
               To know the parameters of each initializer, please refer
               to PyTorch's documentation.
        :type initializer: str
        :param reduce_output: defines how to reduce the output tensor of
               the convolutional layers along the `s` sequence length
               dimension if the rank of the tensor is greater than 2.
               Available values are: `sum`, `mean` or `avg`, `max`, `concat`
               (concatenates along the first dimension), `last` (returns
               the last vector of the first dimension) and `None` or `null`
               (which does not reduce and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

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
                    "filter_size": 7,
                    "pool_size": 3,
                },
                {
                    "filter_size": 7,
                    "pool_size": 3,
                },
                {
                    "filter_size": 3,
                    "pool_size": None,
                },
                {
                    "filter_size": 3,
                    "pool_size": None,
                },
                {
                    "filter_size": 3,
                    "pool_size": None,
                },
                {
                    "filter_size": 3,
                    "pool_size": 3,
                },
            ]
            self.num_conv_layers = 6
        else:
            raise ValueError("Invalid layer parametrization, use either conv_layers or " "num_conv_layers")

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [{"output_size": 512}, {"output_size": 256}]
            num_fc_layers = 2
        elif fc_layers is not None and num_fc_layers is not None:
            raise ValueError("Invalid layer parametrization, use either fc_layers or " "num_fc_layers only. Not both.")

        self.max_sequence_length = max_sequence_length
        self.num_filters = num_filters
        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )

        logger.debug("  Conv1DStack")
        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else embedding_size
        self.conv1d_stack = Conv1DStack(
            in_channels=in_channels,
            max_sequence_length=max_sequence_length,
            layers=self.conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_strides=strides,
            default_padding=padding,
            default_dilation_rate=dilation_rate,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_strides=pool_strides,
            default_pool_padding=pool_padding,
        )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=self.conv1d_stack.output_shape[-2],
            encoding_size=self.conv1d_stack.output_shape[-1],
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=activation,
                default_dropout=dropout,
            )

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return StackedCNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return self.conv1d_stack.output_shape
        return self.fc_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.conv1d_stack(hidden, mask=mask)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        # no reduction: hidden [batch_size, seq_size, num_filters]
        # with reduction: hidden [batch_size, output_size]
        return {ENCODER_OUTPUT: hidden}


@DeveloperAPI
@register_sequence_encoder("stacked_parallel_cnn")
@register_encoder("stacked_parallel_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedParallelCNN(SequenceEncoder):
    """Stacked Parallel CNN encoder combining both parallel and stacked convolutions.

    Embeds the input sequence, then passes it through a stack of parallel convolutional
    layers. At each level in the stack, multiple parallel 1D convolutions with different
    filter sizes are applied and their outputs concatenated (like ParallelCNN). These
    stacked parallel blocks are applied sequentially (like StackedCNN), combining the
    benefits of both approaches: multi-scale pattern detection at each level with
    hierarchical composition across levels.

    Use when: both multi-scale n-gram detection AND hierarchical feature composition are
    needed. This is the most expressive CNN-based sequence encoder but also the most
    computationally expensive.

    Alternatives: ParallelCNN (simpler, single level of parallel convolutions),
    StackedCNN (simpler, single filter size per level but deeper),
    StackedTransformer (when long-range dependencies matter more than local patterns).

    Performance: O(n * S * P * k * f) where n is sequence length, S is num stacked layers,
    P is num parallel filters per layer, k is filter size, f is num_filters. Higher memory
    usage than ParallelCNN or StackedCNN alone.
    """

    def __init__(
        self,
        should_embed=True,
        vocab=None,
        representation="dense",
        embedding_size=256,
        max_sequence_length=None,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        stacked_layers=None,
        num_stacked_layers=None,
        filter_size=3,
        num_filters=256,
        pool_function="max",
        pool_size=None,
        fc_layers=None,
        num_fc_layers=None,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        activation="relu",
        dropout=0,
        reduce_output="max",
        encoder_config=None,
        **kwargs,
    ):
        # todo: review docstring
        """
        :param should_embed: If True the input sequence is expected
               to be made of integers and will be mapped into embeddings
        :type should_embed: Boolean
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: Str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
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
               `filter_size`, `num_filters`, `pool_size`, `norm` and
               `activation`. If any of those values
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
               for each layer are: `output_size`, `norm` and `activation`.
               If any of those values is missing from
               the dictionary, the default one specified as a parameter of
               the encoder will be used instead. If both `fc_layers` and
               `num_fc_layers` are `None`, a default list will be assigned
               to `fc_layers` with the value
               `[{output_size: 512}, {output_size: 256}]`
               (only applies if `reduce_output` is not `None`).
        :type fc_layers: List
        :param num_fc_layers: if `fc_layers` is `None`, this is the number
               of stacked fully connected layers (only applies if
               `reduce_output` is not `None`).
        :type num_fc_layers: Integer
        :param output_size: if a `output_size` is not already specified in
               `fc_layers` this is the default `output_size` that will be used
               for each layer. It indicates the size of the output
               of a fully connected layer.
        :type output_size: Integer
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
               `xavier_uniform`. Options are: `constant`, `identity`,
               `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
               `truncated_normal`, `variance_scaling`, `xavier_normal`,
               `xavier_uniform`, `xavier_normal`,
               `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
               Alternatively it is possible to specify a dictionary with
               a key `type` that identifies the type of initializer and
               other keys for its parameters,
               e.g. `{type: normal, mean: 0, stddev: 0}`.
               To know the parameters of each initializer, please refer
               to PyTorch's documentation.
        :type initializer: str
        :param reduce_output: defines how to reduce the output tensor of
               the convolutional layers along the `s` sequence length
               dimension if the rank of the tensor is greater than 2.
               Available values are: `sum`, `mean` or `avg`, `max`, `concat`
               (concatenates along the first dimension), `last` (returns
               the last vector of the first dimension) and `None` or `null`
               (which does not reduce and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size

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
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
            ]
            self.num_stacked_layers = 6
        else:
            raise ValueError("Invalid layer parametrization, use either stacked_layers or" " num_stacked_layers")

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if fc_layers is None and num_fc_layers is None:
            # use default layers with varying filter sizes
            fc_layers = [{"output_size": 512}, {"output_size": 256}]
            num_fc_layers = 2
        elif fc_layers is not None and num_fc_layers is not None:
            raise ValueError("Invalid layer parametrization, use either fc_layers or " "num_fc_layers only. Not both.")

        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )

        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else embedding_size
        logger.debug("  ParallelConv1DStack")
        self.parallel_conv1d_stack = ParallelConv1DStack(
            in_channels=in_channels,
            stacked_layers=self.stacked_layers,
            max_sequence_length=max_sequence_length,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=activation,
            default_dropout=dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
        )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=self.parallel_conv1d_stack.output_shape[-2],
            encoding_size=self.parallel_conv1d_stack.output_shape[-1],
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=activation,
                default_dropout=dropout,
            )

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return StackedParallelCNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.parallel_conv1d_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.parallel_conv1d_stack(hidden, mask=mask)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        # no reduction: hidden [batch_size, seq_size, num_filter]
        # with reduction: hidden [batch_size, output_size]
        return {ENCODER_OUTPUT: hidden}


@DeveloperAPI
@register_sequence_encoder("rnn")
@register_encoder("rnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedRNN(SequenceEncoder):
    """Stacked recurrent neural network encoder (RNN/LSTM/GRU).

    Embeds the input sequence, then passes it through one or more stacked recurrent layers.
    Supports RNN, LSTM, and GRU cell types, with optional bidirectionality. The recurrent
    layers process the sequence step by step, maintaining a hidden state that captures
    sequential dependencies. Returns both the sequence of hidden states and the final state.

    Use when: sequential order and long-range dependencies are important. LSTM and GRU cells
    handle vanishing gradients better than vanilla RNN. Bidirectional mode captures both
    forward and backward context. Good for tasks where the meaning depends on token order
    (e.g., named entity recognition, sequence labeling).

    Alternatives: ParallelCNN/StackedCNN (faster, but only capture local patterns),
    StackedTransformer (captures long-range dependencies without sequential bottleneck),
    StackedCNNRNN (CNN for local features + RNN for sequential modeling).

    Performance: O(n * L * h) where n is sequence length, L is num layers, h is state size.
    Sequential computation (cannot parallelize across time steps). LSTM/GRU use ~4x/3x the
    parameters of vanilla RNN respectively. Bidirectional doubles parameters and compute.
    """

    def __init__(
        self,
        should_embed=True,
        vocab=None,
        representation="dense",
        embedding_size=256,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        num_layers=1,
        max_sequence_length=None,
        state_size=256,
        cell_type="rnn",
        bidirectional=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        unit_forget_bias=True,
        recurrent_initializer="orthogonal",
        dropout=0.0,
        recurrent_dropout=0.0,
        fc_layers=None,
        num_fc_layers=0,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        fc_activation="relu",
        fc_dropout=0,
        reduce_output="last",
        encoder_config=None,
        **kwargs,
    ):
        # todo: fix up docstring
        """
        :param should_embed: If True the input sequence is expected
               to be made of integers and will be mapped into embeddings
        :type should_embed: Boolean
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: Str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
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
               Available values are: `rnn`, `lstm`, `gru`.
               For reference about the differences between the cells please
               refer to PyTorch's documentation.
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
        :param recurrent_dropout: Dropout rate for the recurrent stack.
        :type recurrent_dropout: float
        :param initializer: the initializer to use. If `None` it uses
               `xavier_uniform`. Options are: `constant`, `identity`,
               `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
               `truncated_normal`, `variance_scaling`, `xavier_normal`,
               `xavier_uniform`, `xavier_normal`,
               `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
               Alternatively it is possible to specify a dictionary with
               a key `type` that identifies the type of initializer and
               other keys for its parameters,
               e.g. `{type: normal, mean: 0, stddev: 0}`.
               To know the parameters of each initializer, please refer
               to PyTorch's documentation.
        :type initializer: str
        :param reduce_output: defines how to reduce the output tensor of
               the convolutional layers along the `s` sequence length
               dimension if the rank of the tensor is greater than 2.
               Available values are: `sum`, `mean` or `avg`, `max`, `concat`
               (concatenates along the first dimension), `last` (returns
               the last vector of the first dimension) and `None` or `null`
               (which does not reduce and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.max_sequence_length = max_sequence_length
        self.hidden_size = state_size
        self.embedding_size = embedding_size

        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )

        logger.debug("  RecurrentStack")
        input_size = self.embed_sequence.output_shape[-1] if self.should_embed else embedding_size
        self.recurrent_stack = RecurrentStack(
            input_size=input_size,
            hidden_size=state_size,
            cell_type=cell_type,
            max_sequence_length=max_sequence_length,
            num_layers=num_layers,
            bidirectional=bidirectional,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
            weights_initializer=weights_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dropout=recurrent_dropout,
        )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=self.recurrent_stack.output_shape[-2],
            encoding_size=self.recurrent_stack.output_shape[-1],  # state_size
        )
        if self.reduce_output is None:
            self.supports_masking = True
        else:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return StackedRNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.recurrent_stack.output_shape

    def input_dtype(self):
        return torch.int32

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Recurrent Layers ================
        hidden, final_state = self.recurrent_stack(hidden, mask=mask)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        return {ENCODER_OUTPUT: hidden, ENCODER_OUTPUT_STATE: final_state}


@DeveloperAPI
@register_sequence_encoder("cnnrnn")
@register_encoder("cnnrnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedCNNRNN(SequenceEncoder):
    """Hybrid CNN-RNN encoder: convolutional layers followed by recurrent layers.

    Embeds the input sequence, passes it through a stack of 1D convolutional layers
    (with optional pooling) to extract local features and reduce sequence length, then
    feeds the result through stacked recurrent layers (RNN/LSTM/GRU) to capture sequential
    dependencies. This two-stage approach combines the strengths of both architectures:
    CNNs efficiently extract local n-gram features while RNNs model long-range sequential
    dependencies over the CNN-reduced sequence.

    Use when: both local pattern extraction AND sequential dependencies are important.
    The CNN layers reduce the sequence length before the RNN, making the RNN more efficient
    on long sequences. Good for tasks like document classification where local features
    need to be composed sequentially.

    Alternatives: StackedRNN (skip CNN, use RNN directly on embeddings),
    StackedCNN (skip RNN, use only convolutions),
    StackedTransformer (attention-based alternative to RNN for long-range dependencies).

    Performance: O(n * k * f) for CNN + O(n' * L * h) for RNN, where n' is the reduced
    sequence length after CNN pooling. The CNN pooling makes the RNN portion faster than
    using RNN alone on the full sequence.
    """

    def __init__(
        self,
        should_embed=True,
        vocab=None,
        max_sequence_length=None,
        representation="dense",
        embedding_size=256,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        conv_layers=None,
        num_conv_layers=None,
        num_filters=256,
        filter_size=5,
        strides=1,
        padding="same",
        dilation_rate=1,
        conv_activation="relu",
        conv_dropout=0.0,
        pool_function="max",
        pool_size=2,
        pool_strides=None,
        pool_padding="same",
        num_rec_layers=1,
        state_size=256,
        cell_type="rnn",
        bidirectional=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        unit_forget_bias=True,
        recurrent_initializer="orthogonal",
        dropout=0.0,
        recurrent_dropout=0.0,
        fc_layers=None,
        num_fc_layers=0,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        fc_activation="relu",
        fc_dropout=0,
        reduce_output="last",
        encoder_config=None,
        **kwargs,
    ):
        # todo: fix up docstring
        """
        :param should_embed: If True the input sequence is expected
               to be made of integers and will be mapped into embeddings
        :type should_embed: Boolean
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: Str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
               on GPU memory if a GPU is used, as it allows
               for faster access, but in some cases the embedding matrix
               may be really big and this parameter forces the placement
               of the embedding matrix in regular memroy and the CPU is used
               to resolve them, slightly slowing down the process
               as a result of data transfer between CPU and GPU memory.
        :param num_layers: the number of stacked recurrent layers.
        :type num_layers: Integer
        :param cell_type: the type of recurrent cell to use.
               Available values are: `rnn`, `lstm`, `gru`.
               For reference about the differences between the cells please
               refer to PyTorch's documentation.
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
        :param recurrent_dropout: Dropout rate for the recurrent stack.
        :type recurrent_dropout: float
        :param initializer: the initializer to use. If `None` it uses
               `xavier_uniform`. Options are: `constant`, `identity`,
               `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
               `truncated_normal`, `variance_scaling`, `xavier_normal`,
               `xavier_uniform`, `xavier_normal`,
               `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
               Alternatively it is possible to specify a dictionary with
               a key `type` that identifies the type of initializer and
               other keys for its parameters,
               e.g. `{type: normal, mean: 0, stddev: 0}`.
               To know the parameters of each initializer, please refer
               to PyTorch's documentation.
        :type initializer: str
        :param reduce_output: defines how to reduce the output tensor of
               the convolutional layers along the `s` sequence length
               dimension if the rank of the tensor is greater than 2.
               Available values are: `sum`, `mean` or `avg`, `max`, `concat`
               (concatenates along the first dimension), `last` (returns
               the last vector of the first dimension) and `None` or `null`
               (which does not reduce and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

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
            self.conv_layers = [{"pool_size": 3}, {"pool_size": None}]
            self.num_conv_layers = 2
        else:
            raise ValueError("Invalid layer parametrization, use either conv_layers or " "num_conv_layers")

        self.max_sequence_length = max_sequence_length
        self.should_embed = should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )

        logger.debug("  Conv1DStack")
        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else embedding_size
        self.conv1d_stack = Conv1DStack(
            in_channels=in_channels,
            max_sequence_length=max_sequence_length,
            layers=self.conv_layers,
            default_num_filters=num_filters,
            default_filter_size=filter_size,
            default_strides=strides,
            default_padding=padding,
            default_dilation_rate=dilation_rate,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=norm,
            default_norm_params=norm_params,
            default_activation=conv_activation,
            default_dropout=conv_dropout,
            default_pool_function=pool_function,
            default_pool_size=pool_size,
            default_pool_strides=pool_strides,
            default_pool_padding=pool_padding,
        )

        logger.debug("  RecurrentStack")
        self.recurrent_stack = RecurrentStack(
            input_size=self.conv1d_stack.output_shape[1],
            hidden_size=state_size,
            max_sequence_length=self.conv1d_stack.output_shape[0],
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
            dropout=recurrent_dropout,
        )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=self.recurrent_stack.output_shape[-2],
            encoding_size=self.recurrent_stack.output_shape[-1],  # State size
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return StackedCNNRNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.recurrent_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)

        # shape=(?, sequence_length, embedding_size)
        hidden = embedded_sequence

        # ================ Conv Layers ================
        hidden = self.conv1d_stack(hidden, mask=mask)

        # ================ Recurrent Layers ================
        hidden, final_state = self.recurrent_stack(hidden)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        # no reduction: hidden [batch_size, seq_size, state_size]
        # with reduction: hidden [batch_size, seq_size, output_size]
        # final_state: if rnn/gru [batch_size, state_size]
        #              lstm ([batch_size, state_size], [batch_size, state_size])
        return {ENCODER_OUTPUT: hidden, ENCODER_OUTPUT_STATE: final_state}


@DeveloperAPI
@register_sequence_encoder("transformer")
@register_encoder("transformer", [SEQUENCE, TEXT, TIMESERIES])
class StackedTransformer(SequenceEncoder):
    """Stacked Transformer encoder using multi-head self-attention.

    Embeds the input sequence with token and positional embeddings, then passes it through
    a stack of Transformer blocks. Each block consists of multi-head self-attention followed
    by a position-wise feedforward network, with residual connections and layer normalization.
    This is the encoder architecture from "Attention Is All You Need" (Vaswani et al., 2017).

    Reference: Vaswani, A., et al. (2017). Attention Is All You Need.
    https://arxiv.org/abs/1706.03762

    Use when: long-range dependencies between tokens are important and the sequence is not
    excessively long. Self-attention allows every position to attend to every other position,
    capturing global context. Best for medium-length sequences (up to ~512-1024 tokens).

    Alternatives: StackedRNN (lower memory for very long sequences, sequential processing),
    ParallelCNN/StackedCNN (faster for local pattern extraction only),
    MambaEncoder (linear-time alternative for very long sequences).

    Performance: O(n^2 * d) time and O(n^2) memory for self-attention, where n is sequence
    length and d is hidden size. Fully parallelizable across positions (unlike RNNs).
    Memory can be a bottleneck for long sequences due to quadratic attention.
    """

    def __init__(
        self,
        max_sequence_length,
        should_embed=True,
        vocab=None,
        representation="dense",
        embedding_size=256,
        embeddings_trainable=True,
        pretrained_embeddings=None,
        embeddings_on_cpu=False,
        num_layers=1,
        hidden_size=256,
        num_heads=8,
        transformer_output_size=256,
        dropout=0.1,
        fc_layers=None,
        num_fc_layers=0,
        output_size=256,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        norm=None,
        norm_params=None,
        fc_activation="relu",
        fc_dropout=0,
        use_rope=False,
        reduce_output="last",
        encoder_config=None,
        **kwargs,
    ):
        # todo: update docstring as needed
        """
        :param should_embed: If True the input sequence is expected
               to be made of integers and will be mapped into embeddings
        :type should_embed: Boolean
        :param vocab: Vocabulary of the input feature to encode
        :type vocab: List
        :param representation: the possible values are `dense` and `sparse`.
               `dense` means the embeddings are initialized randomly,
               `sparse` means they are initialized to be one-hot encodings.
        :type representation: Str (one of 'dense' or 'sparse')
        :param embedding_size: it is the maximum embedding size, the actual
               size will be `min(vocabulary_size, embedding_size)`
               for `dense` representations and exactly `vocabulary_size`
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
        :param embeddings_on_cpu: by default embeddings matrices are stored
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
               Available values are: `rnn`, `lstm`, `lstm_block`, `lstm`,
               `ln`, `lstm_cudnn`, `gru`, `gru_block`, `gru_cudnn`.
               For reference about the differences between the cells please
               refer to PyTorch's documentation. We suggest to use the
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
               `xavier_uniform`. Options are: `constant`, `identity`,
               `zeros`, `ones`, `orthogonal`, `normal`, `uniform`,
               `truncated_normal`, `variance_scaling`, `xavier_normal`,
               `xavier_uniform`, `xavier_normal`,
               `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.
               Alternatively it is possible to specify a dictionary with
               a key `type` that identifies the type of initializer and
               other keys for its parameters,
               e.g. `{type: normal, mean: 0, stddev: 0}`.
               To know the parameters of each initializer, please refer
               to PyTorch's documentation.
        :type initializer: str
        :param reduce_output: defines how to reduce the output tensor of
               the convolutional layers along the `s` sequence length
               dimension if the rank of the tensor is greater than 2.
               Available values are: `sum`, `mean` or `avg`, `max`, `concat`
               (concatenates along the first dimension), `last` (returns
               the last vector of the first dimension) and `None` or `null`
               (which does not reduce and returns the full tensor).
        :type reduce_output: str
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.max_sequence_length = max_sequence_length

        self.should_embed = should_embed
        self.should_project = False
        self.embed_sequence = None
        self.use_rope = use_rope

        if self.should_embed:
            if use_rope:
                logger.debug("  EmbedSequence (token-only, RoPE handles positions)")
                self.embed_sequence = EmbedSequence(
                    vocab=vocab,
                    embedding_size=embedding_size,
                    max_sequence_length=max_sequence_length,
                    representation=representation,
                    embeddings_trainable=embeddings_trainable,
                    pretrained_embeddings=pretrained_embeddings,
                    embeddings_on_cpu=embeddings_on_cpu,
                    dropout=dropout,
                    embedding_initializer=weights_initializer,
                )
            else:
                logger.debug("  EmbedSequence")
                self.embed_sequence = TokenAndPositionEmbedding(
                    max_sequence_length=max_sequence_length,
                    vocab=vocab,
                    embedding_size=embedding_size,
                    representation=representation,
                    embeddings_trainable=embeddings_trainable,
                    pretrained_embeddings=pretrained_embeddings,
                    embeddings_on_cpu=embeddings_on_cpu,
                    dropout=dropout,
                    embedding_initializer=weights_initializer,
                )
            # If vocab size is smaller than embedding size, embedding layer will use len(vocab) as embedding_size.
            used_embedding_size = self.embed_sequence.output_shape[-1]
            if used_embedding_size != hidden_size:
                logger.debug("  project_to_embed_size")
                self.project_to_hidden_size = nn.Linear(self.embed_sequence.output_shape[-1], hidden_size)
                self.should_project = True
        else:
            logger.debug("  project_to_embed_size")
            self.project_to_hidden_size = nn.Linear(embedding_size, hidden_size)
            self.should_project = True

        logger.debug("  TransformerStack")
        self.transformer_stack = TransformerStack(
            input_size=hidden_size,
            max_sequence_length=max_sequence_length,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_size=transformer_output_size,
            num_layers=num_layers,
            dropout=dropout,
            use_rope=use_rope,
        )

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=self.transformer_stack.output_shape[-2],
            encoding_size=self.transformer_stack.output_shape[-1],  # hidden_size
        )
        if self.reduce_output is None:
            self.supports_masking = True
        else:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return StackedTransformerConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.transformer_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)

        # shape=(?, sequence_length, embedding_size)
        if self.should_project:
            hidden = self.project_to_hidden_size(embedded_sequence)
        else:
            hidden = embedded_sequence
        # shape=(?, sequence_length, hidden)

        # ================ Transformer Layers ================
        hidden = self.transformer_stack(hidden, mask=mask)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        return {ENCODER_OUTPUT: hidden}


class GatedConvSSMLayer(nn.Module):
    """A single SSM-inspired layer using gated depthwise convolution.

    Approximates the selective state space model mechanism using:
    1. Linear projection to an expanded inner dimension
    2. Depthwise 1D convolution for local context mixing
    3. SiLU activation gating (like Mamba's selective mechanism)
    4. Linear projection back to model dimension

    This captures the key insight of Mamba (local convolution + gating) without
    requiring the mamba_ssm CUDA kernels.
    """

    def __init__(self, d_model: int, d_conv: int = 4, expand_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand_factor

        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner, bias=True)
        # Output projection (contract)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)

        # Project and split into two paths (gate and value)
        xz = self.in_proj(x)  # [batch, seq_len, 2 * d_inner]
        x_path, z = xz.chunk(2, dim=-1)  # each [batch, seq_len, d_inner]

        # Depthwise convolution on x path
        x_path = x_path.transpose(1, 2)  # [batch, d_inner, seq_len]
        x_path = self.conv1d(x_path)[:, :, : x.shape[1]]  # Causal: trim to original length
        x_path = x_path.transpose(1, 2)  # [batch, seq_len, d_inner]

        # Gated activation (SiLU gate, like Mamba)
        x_path = torch.nn.functional.silu(x_path) * z

        # Project back
        out = self.out_proj(x_path)
        out = self.dropout(out)
        return out + residual


@DeveloperAPI
@register_encoder("mamba", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class MambaEncoder(SequenceEncoder):
    """State space model (SSM) encoder inspired by Mamba (Gu & Dao, 2024).

    Uses a stack of gated convolutional SSM layers that combine depthwise 1D convolution
    with SiLU-gated projections, capturing the key mechanism of Mamba's selective state
    space model. Each layer uses local convolution for context mixing and gating for
    selective information flow, without requiring the mamba_ssm CUDA kernels.

    For full Mamba support with CUDA-optimized selective scan, install the mamba_ssm
    package (pip install mamba-ssm). This encoder provides a pure-PyTorch approximation
    that captures the architectural pattern.

    Reference: Gu, A. & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces. https://arxiv.org/abs/2312.00752

    Use when: processing very long sequences where Transformer attention is too expensive
    (memory grows linearly instead of quadratically), or when linear-time inference is
    required. Also effective for audio and timeseries data.

    Alternatives: StackedTransformer (better for short/medium sequences where quadratic
    attention is affordable), StackedRNN (simpler recurrent alternative),
    StackedCNN (when only local patterns matter).

    Performance: O(n * d * e) time and O(n * d) memory where n is sequence length,
    d is d_model, e is expand_factor. Linear in sequence length (vs. quadratic for
    Transformers). The gated convolution is fully parallelizable.
    """

    def __init__(
        self,
        max_sequence_length: int = 256,
        should_embed: bool = True,
        vocab=None,
        representation: str = "dense",
        embedding_size: int = 256,
        embeddings_trainable: bool = True,
        pretrained_embeddings=None,
        embeddings_on_cpu: bool = False,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        fc_layers=None,
        num_fc_layers: int = 0,
        output_size: int = 256,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm=None,
        norm_params=None,
        fc_activation: str = "relu",
        fc_dropout: float = 0,
        reduce_output: str = "mean",
        encoder_config=None,
        **kwargs,
    ):
        """
        :param max_sequence_length: Maximum sequence length.
        :param should_embed: If True, input tokens (integers) are mapped to embeddings.
        :param vocab: Vocabulary of the input feature.
        :param representation: 'dense' for learned embeddings, 'sparse' for one-hot.
        :param embedding_size: Size of token embeddings.
        :param d_model: Hidden dimension of the SSM layers.
        :param n_layers: Number of stacked SSM layers.
        :param d_state: State dimension (unused in gated-conv fallback, reserved for full Mamba).
        :param d_conv: Kernel size of the depthwise convolution in each SSM layer.
        :param expand_factor: Expansion factor for the inner dimension of SSM layers.
        :param dropout: Dropout rate.
        :param reduce_output: How to reduce the sequence dimension. Default 'mean'.
        """
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.should_embed = should_embed
        self.should_project = False
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                vocab,
                embedding_size,
                max_sequence_length=max_sequence_length,
                representation=representation,
                embeddings_trainable=embeddings_trainable,
                pretrained_embeddings=pretrained_embeddings,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )
            used_embedding_size = self.embed_sequence.output_shape[-1]
            if used_embedding_size != d_model:
                logger.debug("  project_to_d_model")
                self.project_to_d_model = nn.Linear(used_embedding_size, d_model)
                self.should_project = True
        else:
            if embedding_size != d_model:
                logger.debug("  project_to_d_model")
                self.project_to_d_model = nn.Linear(embedding_size, d_model)
                self.should_project = True

        logger.debug("  SSM Layers")
        self.ssm_layers = nn.ModuleList(
            [
                GatedConvSSMLayer(
                    d_model=d_model,
                    d_conv=d_conv,
                    expand_factor=expand_factor,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.reduce_output = reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=reduce_output,
            max_sequence_length=max_sequence_length,
            encoding_size=d_model,
        )
        if self.reduce_output is None:
            self.supports_masking = True
        else:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=fc_layers,
                num_layers=num_fc_layers,
                default_output_size=output_size,
                default_use_bias=use_bias,
                default_weights_initializer=weights_initializer,
                default_bias_initializer=bias_initializer,
                default_norm=norm,
                default_norm_params=norm_params,
                default_activation=fc_activation,
                default_dropout=fc_dropout,
            )

    @staticmethod
    def get_schema_cls() -> type[SequenceEncoderConfig]:
        return MambaEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return torch.Size([self.max_sequence_length, self.d_model])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused in current SSM implementation)
        """
        # ================ Embeddings ================
        if self.should_embed:
            embedded_sequence = self.embed_sequence(inputs, mask=mask)
        else:
            embedded_sequence = inputs
            while len(embedded_sequence.shape) < 3:
                embedded_sequence = embedded_sequence.unsqueeze(-1)

        # shape=(?, sequence_length, embedding_size)
        if self.should_project:
            hidden = self.project_to_d_model(embedded_sequence)
        else:
            hidden = embedded_sequence

        # ================ SSM Layers ================
        for layer in self.ssm_layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)

        # ================ Sequence Reduction ================
        if self.reduce_output is not None:
            hidden = self.reduce_sequence(hidden)

            # ================ FC Layers ================
            hidden = self.fc_stack(hidden, mask=mask)

        return {ENCODER_OUTPUT: hidden}
