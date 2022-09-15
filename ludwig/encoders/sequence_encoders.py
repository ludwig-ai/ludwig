#! /usr/bin/env python
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
from typing import Optional

import torch
from torch import nn

from ludwig.constants import AUDIO, SEQUENCE, TEXT, TIMESERIES
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder, register_sequence_encoder
from ludwig.modules.attention_modules import TransformerStack
from ludwig.modules.convolutional_modules import Conv1DStack, ParallelConv1D, ParallelConv1DStack
from ludwig.modules.embedding_modules import EmbedSequence, TokenAndPositionEmbedding
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.recurrent_modules import RecurrentStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.encoders.sequence_encoders import (
    ParallelCNNConfig,
    SequenceEmbedConfig,
    SequencePassthroughConfig,
    StackedCNNConfig,
    StackedCNNRNNConfig,
    StackedParallelCNNConfig,
    StackedRNNConfig,
    StackedTransformerConfig,
)

logger = logging.getLogger(__name__)


@register_encoder("passthrough", [SEQUENCE, TEXT, TIMESERIES])
class SequencePassthroughEncoder(Encoder):
    def __init__(self, encoder_config: SequencePassthroughConfig = SequencePassthroughConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=encoder_config.reduce_output,
            max_sequence_length=encoder_config.max_sequence_length,
            encoding_size=encoder_config.encoding_size,
        )
        if self.reduce_output is None:
            self.supports_masking = True

    def forward(self, input_sequence, mask=None):
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

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return SequencePassthroughConfig


@register_encoder("embed", [SEQUENCE, TEXT])
class SequenceEmbedEncoder(Encoder):
    def __init__(self, encoder_config: SequenceEmbedConfig = SequenceEmbedConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")
        self.embedding_size = encoder_config.embedding_size
        self.max_sequence_length = encoder_config.max_sequence_length

        self.reduce_output = encoder_config.reduce_output
        if self.reduce_output is None:
            self.supports_masking = True

        logger.debug("  EmbedSequence")
        self.embed_sequence = EmbedSequence(
            encoder_config.vocab,
            encoder_config.embedding_size,
            max_sequence_length=encoder_config.max_sequence_length,
            representation=encoder_config.representation,
            embeddings_trainable=encoder_config.embeddings_trainable,
            pretrained_embeddings=encoder_config.pretrained_embeddings,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        self.reduce_sequence = SequenceReducer(
            reduce_mode=encoder_config.reduce_output,
            max_sequence_length=encoder_config.max_sequence_length,
            encoding_size=self.embed_sequence.output_shape[-1],
        )

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        :param inputs: The input sequence fed into the encoder.
               Shape: [batch x sequence length], type torch.int32
        :param mask: Input mask (unused, not yet implemented in EmbedSequence)
        """
        embedded_sequence = self.embed_sequence(inputs, mask=mask)
        hidden = self.reduce_sequence(embedded_sequence)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return SequenceEmbedConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return self.reduce_sequence.output_shape


@register_sequence_encoder("parallel_cnn")
@register_encoder("parallel_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class ParallelCNN(Encoder):
    def __init__(self, encoder_config: ParallelCNNConfig = ParallelCNNConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        self.max_sequence_length = encoder_config.max_sequence_length

        if encoder_config.conv_layers is not None and encoder_config.num_conv_layers is None:
            # use custom-defined layers
            self.conv_layers = encoder_config.conv_layers
            self.num_conv_layers = len(encoder_config.conv_layers)
        elif encoder_config.conv_layers is None and encoder_config.num_conv_layers is not None:
            # generate num_conv_layers with default parameters
            self.conv_layers = None
            self.num_conv_layers = encoder_config.num_conv_layers
        elif encoder_config.conv_layers is None and encoder_config.num_conv_layers is None:
            # use default layers with varying filter sizes
            self.conv_layers = [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}]
            self.num_conv_layers = 4
        else:
            raise ValueError("Invalid layer parametrization, use either conv_layers or num_conv_layers")

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if encoder_config.fc_layers is None and encoder_config.num_fc_layers is None:
            # use default layers with varying filter sizes
            encoder_config.fc_layers = [{"output_size": 512}, {"output_size": 256}]
        elif encoder_config.fc_layers is not None and encoder_config.num_fc_layers is not None:
            raise ValueError("Invalid layer parametrization, use either fc_layers or num_fc_layers only. Not both.")

        self.should_embed = encoder_config.should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                encoder_config.vocab,
                encoder_config.embedding_size,
                max_sequence_length=encoder_config.max_sequence_length,
                representation=encoder_config.representation,
                embeddings_trainable=encoder_config.embeddings_trainable,
                pretrained_embeddings=encoder_config.pretrained_embeddings,
                embeddings_on_cpu=encoder_config.embeddings_on_cpu,
                dropout=encoder_config.dropout,
                embedding_initializer=encoder_config.weights_initializer,
            )

        logger.debug("  ParallelConv1D")
        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else encoder_config.embedding_size
        self.parallel_conv1d = ParallelConv1D(
            in_channels=in_channels,
            max_sequence_length=self.max_sequence_length,
            layers=self.conv_layers,
            default_num_filters=encoder_config.num_filters,
            default_filter_size=encoder_config.filter_size,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=encoder_config.norm,
            default_norm_params=encoder_config.norm_params,
            default_activation=encoder_config.activation,
            default_dropout=encoder_config.dropout,
            default_pool_function=encoder_config.pool_function,
            default_pool_size=encoder_config.pool_size,
            default_pool_padding="same",
        )

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=encoder_config.reduce_output,
            max_sequence_length=self.max_sequence_length,
            encoding_size=self.parallel_conv1d.output_shape[-1],
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=encoder_config.fc_layers,
                num_layers=encoder_config.num_fc_layers,
                default_output_size=encoder_config.output_size,
                default_use_bias=encoder_config.use_bias,
                default_weights_initializer=encoder_config.weights_initializer,
                default_bias_initializer=encoder_config.bias_initializer,
                default_norm=encoder_config.norm,
                default_norm_params=encoder_config.norm_params,
                default_activation=encoder_config.activation,
                default_dropout=encoder_config.dropout,
            )

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
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

        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return ParallelCNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.parallel_conv1d.output_shape


@register_sequence_encoder("stacked_cnn")
@register_encoder("stacked_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedCNN(Encoder):
    def __init__(self, encoder_config: StackedCNNConfig = StackedCNNConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        if encoder_config.conv_layers is not None and encoder_config.num_conv_layers is None:
            # use custom-defined layers
            self.conv_layers = encoder_config.conv_layers
            self.num_conv_layers = len(encoder_config.conv_layers)
        elif encoder_config.conv_layers is None and encoder_config.num_conv_layers is not None:
            # generate num_conv_layers with default parameters
            self.conv_layers = None
            self.num_conv_layers = encoder_config.num_conv_layers
        elif encoder_config.conv_layers is None and encoder_config.num_conv_layers is None:
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
        if encoder_config.fc_layers is None and encoder_config.num_fc_layers is None:
            # use default layers with varying filter sizes
            encoder_config.fc_layers = [{"output_size": 512}, {"output_size": 256}]
        elif encoder_config.fc_layers is not None and encoder_config.num_fc_layers is not None:
            raise ValueError("Invalid layer parametrization, use either fc_layers or " "num_fc_layers only. Not both.")

        self.max_sequence_length = encoder_config.max_sequence_length
        self.num_filters = encoder_config.num_filters
        self.should_embed = encoder_config.should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                encoder_config.vocab,
                encoder_config.embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=encoder_config.representation,
                embeddings_trainable=encoder_config.embeddings_trainable,
                pretrained_embeddings=encoder_config.pretrained_embeddings,
                embeddings_on_cpu=encoder_config.embeddings_on_cpu,
                dropout=encoder_config.dropout,
                embedding_initializer=encoder_config.weights_initializer,
            )

        logger.debug("  Conv1DStack")
        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else encoder_config.embedding_size
        self.conv1d_stack = Conv1DStack(
            in_channels=in_channels,
            max_sequence_length=self.max_sequence_length,
            layers=self.conv_layers,
            num_layers=self.num_conv_layers,
            default_num_filters=encoder_config.num_filters,
            default_filter_size=encoder_config.filter_size,
            default_strides=encoder_config.strides,
            default_padding=encoder_config.padding,
            default_dilation_rate=encoder_config.dilation_rate,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=encoder_config.norm,
            default_norm_params=encoder_config.norm_params,
            default_activation=encoder_config.activation,
            default_dropout=encoder_config.dropout,
            default_pool_function=encoder_config.pool_function,
            default_pool_size=encoder_config.pool_size,
            default_pool_strides=encoder_config.pool_strides,
            default_pool_padding=encoder_config.pool_padding,
        )

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=self.reduce_output,
            max_sequence_length=self.conv1d_stack.output_shape[-2],
            encoding_size=self.conv1d_stack.output_shape[-1],
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=encoder_config.fc_layers,
                num_layers=encoder_config.num_fc_layers,
                default_output_size=encoder_config.output_size,
                default_use_bias=encoder_config.use_bias,
                default_weights_initializer=encoder_config.weights_initializer,
                default_bias_initializer=encoder_config.bias_initializer,
                default_norm=encoder_config.norm,
                default_norm_params=encoder_config.norm_params,
                default_activation=encoder_config.activation,
                default_dropout=encoder_config.dropout,
            )

    @staticmethod
    def get_schema_cls():
        return StackedCNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is None:
            return self.conv1d_stack.output_shape
        return self.fc_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
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
        return {"encoder_output": hidden}


@register_sequence_encoder("stacked_parallel_cnn")
@register_encoder("stacked_parallel_cnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedParallelCNN(Encoder):
    def __init__(self, encoder_config: StackedParallelCNNConfig = StackedParallelCNNConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        self.max_sequence_length = encoder_config.max_sequence_length
        self.embedding_size = encoder_config.embedding_size

        if encoder_config.stacked_layers is not None and encoder_config.num_stacked_layers is None:
            # use custom-defined layers
            self.stacked_layers = encoder_config.stacked_layers
        elif encoder_config.stacked_layers is None and encoder_config.num_stacked_layers is not None:
            # generate num_conv_layers with default parameters
            self.stacked_layers = [
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
            ] * encoder_config.num_stacked_layers
        elif encoder_config.stacked_layers is None and encoder_config.num_stacked_layers is None:
            # use default layers with varying filter sizes
            self.stacked_layers = [
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
                [{"filter_size": 2}, {"filter_size": 3}, {"filter_size": 4}, {"filter_size": 5}],
            ]
        else:
            raise ValueError("Invalid layer parametrization, use either stacked_layers or" " num_stacked_layers")

        # The user is expected to provide fc_layers or num_fc_layers
        # The following logic handles the case where the user either provides
        # both or neither.
        if encoder_config.fc_layers is None and encoder_config.num_fc_layers is None:
            # use default layers with varying filter sizes
            encoder_config.fc_layers = [{"output_size": 512}, {"output_size": 256}]
        elif encoder_config.fc_layers is not None and encoder_config.num_fc_layers is not None:
            raise ValueError("Invalid layer parametrization, use either fc_layers or " "num_fc_layers only. Not both.")

        self.should_embed = encoder_config.should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                encoder_config.vocab,
                encoder_config.embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=encoder_config.representation,
                embeddings_trainable=encoder_config.embeddings_trainable,
                pretrained_embeddings=encoder_config.pretrained_embeddings,
                embeddings_on_cpu=encoder_config.embeddings_on_cpu,
                dropout=encoder_config.dropout,
                embedding_initializer=encoder_config.weights_initializer,
            )

        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else encoder_config.embedding_size
        logger.debug("  ParallelConv1DStack")
        self.parallel_conv1d_stack = ParallelConv1DStack(
            in_channels=in_channels,
            stacked_layers=self.stacked_layers,
            max_sequence_length=self.max_sequence_length,
            default_num_filters=encoder_config.num_filters,
            default_filter_size=encoder_config.filter_size,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=encoder_config.norm,
            default_norm_params=encoder_config.norm_params,
            default_activation=encoder_config.activation,
            default_dropout=encoder_config.dropout,
            default_pool_function=encoder_config.pool_function,
            default_pool_size=encoder_config.pool_size,
        )

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=self.reduce_output,
            max_sequence_length=self.parallel_conv1d_stack.output_shape[-2],
            encoding_size=self.parallel_conv1d_stack.output_shape[-1],
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=encoder_config.fc_layers,
                num_layers=encoder_config.num_fc_layers,
                default_output_size=encoder_config.output_size,
                default_use_bias=encoder_config.use_bias,
                default_weights_initializer=encoder_config.weights_initializer,
                default_bias_initializer=encoder_config.bias_initializer,
                default_norm=encoder_config.norm,
                default_norm_params=encoder_config.norm_params,
                default_activation=encoder_config.activation,
                default_dropout=encoder_config.dropout,
            )

    @staticmethod
    def get_schema_cls():
        return StackedParallelCNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.parallel_conv1d_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
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
        return {"encoder_output": hidden}


@register_sequence_encoder("rnn")
@register_encoder("rnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedRNN(Encoder):
    def __init__(self, encoder_config: StackedRNNConfig = StackedRNNConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        self.max_sequence_length = encoder_config.max_sequence_length
        self.hidden_size = encoder_config.state_size
        self.embedding_size = encoder_config.embedding_size

        self.should_embed = encoder_config.should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                encoder_config.vocab,
                encoder_config.embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=encoder_config.representation,
                embeddings_trainable=encoder_config.embeddings_trainable,
                pretrained_embeddings=encoder_config.pretrained_embeddings,
                embeddings_on_cpu=encoder_config.embeddings_on_cpu,
                dropout=encoder_config.dropout,
                embedding_initializer=encoder_config.weights_initializer,
            )

        logger.debug("  RecurrentStack")
        input_size = self.embed_sequence.output_shape[-1] if self.should_embed else self.embedding_size
        self.recurrent_stack = RecurrentStack(
            input_size=input_size,
            hidden_size=encoder_config.state_size,
            cell_type=encoder_config.cell_type,
            max_sequence_length=encoder_config.max_sequence_length,
            num_layers=encoder_config.num_layers,
            bidirectional=encoder_config.bidirectional,
            activation=encoder_config.activation,
            recurrent_activation=encoder_config.recurrent_activation,
            use_bias=encoder_config.use_bias,
            unit_forget_bias=encoder_config.unit_forget_bias,
            weights_initializer=encoder_config.weights_initializer,
            recurrent_initializer=encoder_config.recurrent_initializer,
            bias_initializer=encoder_config.bias_initializer,
            dropout=encoder_config.recurrent_dropout,
        )

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=self.reduce_output,
            max_sequence_length=self.recurrent_stack.output_shape[-2],
            encoding_size=self.recurrent_stack.output_shape[-1],  # state_size
        )
        if self.reduce_output is None:
            self.supports_masking = True
        else:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=encoder_config.fc_layers,
                num_layers=encoder_config.num_fc_layers,
                default_output_size=encoder_config.output_size,
                default_use_bias=encoder_config.use_bias,
                default_weights_initializer=encoder_config.weights_initializer,
                default_bias_initializer=encoder_config.bias_initializer,
                default_norm=encoder_config.norm,
                default_norm_params=encoder_config.norm_params,
                default_activation=encoder_config.fc_activation,
                default_dropout=encoder_config.fc_dropout,
            )

    @staticmethod
    def get_schema_cls():
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

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
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

        return {"encoder_output": hidden, "encoder_output_state": final_state}


@register_sequence_encoder("cnnrnn")
@register_encoder("cnnrnn", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class StackedCNNRNN(Encoder):
    def __init__(self, encoder_config: StackedCNNRNNConfig = StackedCNNRNNConfig()):
        super().__init__(encoder_config)

        logger.debug(f" {self.name}")

        if encoder_config.conv_layers is not None and encoder_config.num_conv_layers is None:
            # use custom-defined layers
            self.conv_layers = encoder_config.conv_layers
            self.num_conv_layers = len(encoder_config.conv_layers)
        elif encoder_config.conv_layers is None and encoder_config.num_conv_layers is not None:
            # generate num_conv_layers with default parameters
            self.conv_layers = None
            self.num_conv_layers = encoder_config.num_conv_layers
        elif encoder_config.conv_layers is None and encoder_config.num_conv_layers is None:
            # use default layers with varying filter sizes
            self.conv_layers = [{"pool_size": 3}, {"pool_size": None}]
            self.num_conv_layers = 2
        else:
            raise ValueError("Invalid layer parametrization, use either conv_layers or " "num_conv_layers")

        self.max_sequence_length = encoder_config.max_sequence_length
        self.should_embed = encoder_config.should_embed
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = EmbedSequence(
                encoder_config.vocab,
                encoder_config.embedding_size,
                max_sequence_length=self.max_sequence_length,
                representation=encoder_config.representation,
                embeddings_trainable=encoder_config.embeddings_trainable,
                pretrained_embeddings=encoder_config.pretrained_embeddings,
                embeddings_on_cpu=encoder_config.embeddings_on_cpu,
                dropout=encoder_config.dropout,
                embedding_initializer=encoder_config.weights_initializer,
            )

        logger.debug("  Conv1DStack")
        in_channels = self.embed_sequence.output_shape[-1] if self.should_embed else encoder_config.embedding_size
        self.conv1d_stack = Conv1DStack(
            in_channels=in_channels,
            max_sequence_length=self.max_sequence_length,
            layers=self.conv_layers,
            num_layers=self.num_conv_layers,
            default_num_filters=encoder_config.num_filters,
            default_filter_size=encoder_config.filter_size,
            default_strides=encoder_config.strides,
            default_padding=encoder_config.padding,
            default_dilation_rate=encoder_config.dilation_rate,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=encoder_config.norm,
            default_norm_params=encoder_config.norm_params,
            default_activation=encoder_config.conv_activation,
            default_dropout=encoder_config.conv_dropout,
            default_pool_function=encoder_config.pool_function,
            default_pool_size=encoder_config.pool_size,
            default_pool_strides=encoder_config.pool_strides,
            default_pool_padding=encoder_config.pool_padding,
        )

        logger.debug("  RecurrentStack")
        self.recurrent_stack = RecurrentStack(
            input_size=self.conv1d_stack.output_shape[1],
            hidden_size=encoder_config.state_size,
            max_sequence_length=self.conv1d_stack.output_shape[0],
            cell_type=encoder_config.cell_type,
            num_layers=encoder_config.num_rec_layers,
            bidirectional=encoder_config.bidirectional,
            activation=encoder_config.activation,
            recurrent_activation=encoder_config.recurrent_activation,
            use_bias=encoder_config.use_bias,
            unit_forget_bias=encoder_config.unit_forget_bias,
            weights_initializer=encoder_config.weights_initializer,
            recurrent_initializer=encoder_config.recurrent_initializer,
            bias_initializer=encoder_config.bias_initializer,
            dropout=encoder_config.recurrent_dropout,
        )

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=self.reduce_output,
            max_sequence_length=self.recurrent_stack.output_shape[-2],
            encoding_size=self.recurrent_stack.output_shape[-1],  # State size
        )
        if self.reduce_output is not None:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=encoder_config.fc_layers,
                num_layers=encoder_config.num_fc_layers,
                default_output_size=encoder_config.output_size,
                default_use_bias=encoder_config.use_bias,
                default_weights_initializer=encoder_config.weights_initializer,
                default_bias_initializer=encoder_config.bias_initializer,
                default_norm=encoder_config.norm,
                default_norm_params=encoder_config.norm_params,
                default_activation=encoder_config.fc_activation,
                default_dropout=encoder_config.fc_dropout,
            )

    @staticmethod
    def get_schema_cls():
        return StackedCNNRNNConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.recurrent_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
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
        return {"encoder_output": hidden, "encoder_output_state": final_state}


@register_encoder("transformer", [SEQUENCE, TEXT, TIMESERIES])
class StackedTransformer(Encoder):
    def __init__(self, encoder_config: StackedTransformerConfig = StackedTransformerConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        self.max_sequence_length = encoder_config.max_sequence_length

        self.should_embed = encoder_config.should_embed
        self.should_project = False
        self.embed_sequence = None

        if self.should_embed:
            logger.debug("  EmbedSequence")
            self.embed_sequence = TokenAndPositionEmbedding(
                max_sequence_length=self.max_sequence_length,
                vocab=encoder_config.vocab,
                embedding_size=encoder_config.embedding_size,
                representation=encoder_config.representation,
                embeddings_trainable=encoder_config.embeddings_trainable,
                pretrained_embeddings=encoder_config.pretrained_embeddings,
                embeddings_on_cpu=encoder_config.embeddings_on_cpu,
                dropout=encoder_config.dropout,
                embedding_initializer=encoder_config.weights_initializer,
            )
            # If vocab size is smaller than embedding size, embedding layer will use len(vocab) as embedding_size.
            used_embedding_size = self.embed_sequence.output_shape[-1]
            if used_embedding_size != encoder_config.hidden_size:
                logger.debug("  project_to_embed_size")
                self.project_to_hidden_size = nn.Linear(
                    self.embed_sequence.output_shape[-1], encoder_config.hidden_size
                )
                self.should_project = True
        else:
            logger.debug("  project_to_embed_size")
            self.project_to_hidden_size = nn.Linear(1, encoder_config.hidden_size)
            self.should_project = True

        logger.debug("  TransformerStack")
        self.transformer_stack = TransformerStack(
            input_size=encoder_config.hidden_size,
            max_sequence_length=self.max_sequence_length,
            hidden_size=encoder_config.hidden_size,
            num_heads=encoder_config.num_heads,
            output_size=encoder_config.transformer_output_size,
            num_layers=encoder_config.num_layers,
            dropout=encoder_config.dropout,
        )

        self.reduce_output = encoder_config.reduce_output
        self.reduce_sequence = SequenceReducer(
            reduce_mode=self.reduce_output,
            max_sequence_length=self.transformer_stack.output_shape[-2],
            encoding_size=self.transformer_stack.output_shape[-1],  # hidden_size
        )
        if self.reduce_output is None:
            self.supports_masking = True
        else:
            logger.debug("  FCStack")
            self.fc_stack = FCStack(
                self.reduce_sequence.output_shape[-1],
                layers=encoder_config.fc_layers,
                num_layers=encoder_config.num_fc_layers,
                default_output_size=encoder_config.output_size,
                default_use_bias=encoder_config.use_bias,
                default_weights_initializer=encoder_config.weights_initializer,
                default_bias_initializer=encoder_config.bias_initializer,
                default_norm=encoder_config.norm,
                default_norm_params=encoder_config.norm_params,
                default_activation=encoder_config.fc_activation,
                default_dropout=encoder_config.fc_dropout,
            )

    @staticmethod
    def get_schema_cls():
        return StackedTransformerConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output is not None:
            return self.fc_stack.output_shape
        return self.transformer_stack.output_shape

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
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

        return {"encoder_output": hidden}
