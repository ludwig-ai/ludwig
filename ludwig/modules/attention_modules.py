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

import torch
from torch import nn
from torch.nn import functional as F

from ludwig.utils.torch_utils import get_activation, LudwigModule

logger = logging.getLogger(__name__)


class FeedForwardAttentionReducer(LudwigModule):
    def __init__(self, input_size, hidden_size=256, activation="tanh"):
        super().__init__()
        self.fc_layer1 = nn.Linear(input_size, hidden_size)
        self.fc_layer1_activation = get_activation(activation)
        self.fc_layer2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, mask=None):
        # current_inputs shape [b, s, h]
        hidden = self.fc_layer1(inputs)  # [b, s, h']
        hidden = self.fc_layer1_activation(hidden)
        hidden = self.fc_layer2(hidden)  # [b, s, 1]
        attention = F.softmax(hidden, dim=1)
        gated_inputs = torch.sum(attention * inputs, dim=1)
        return gated_inputs  # [b, h]


class MultiHeadSelfAttention(LudwigModule):
    def __init__(self, input_size, hidden_size, num_heads=8):
        super().__init__()
        self.embedding_size = hidden_size
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"When using multi-head attention, `hidden_size` ({hidden_size}), should be divisible by "
                f"`num_heads` ({num_heads}). Please update the `transformer` section of the model config."
            )
        self.projection_dim = hidden_size // num_heads
        self.query_dense = nn.Linear(input_size, hidden_size)
        self.key_dense = nn.Linear(input_size, hidden_size)
        self.value_dense = nn.Linear(input_size, hidden_size)
        self.combine_heads = nn.Linear(hidden_size, hidden_size)

    def attention(self, query, key, value, mask=None):
        score = torch.matmul(query, key.permute(0, 1, 3, 2))
        dim_key = torch.tensor(key.shape[-1]).type(torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        if mask:
            scaled_score = mask * scaled_score
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, inputs, batch_size):
        inputs = torch.reshape(inputs, (batch_size, -1, self.num_heads, self.projection_dim))
        return torch.permute(inputs, (0, 2, 1, 3))

    def forward(self, inputs: torch.Tensor, mask=None):
        # inputs.shape = [batch_size, seq_len, embedding_dim]
        batch_size = inputs.shape[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, h)
        key = self.key_dense(inputs)  # (batch_size, seq_len, h)
        value = self.value_dense(inputs)  # (batch_size, seq_len, h)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        outputs, weights = self.attention(query, key, value, mask=mask)
        outputs = torch.permute(outputs, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, projection_dim)
        concat_outputs = torch.reshape(outputs, (batch_size, -1, self.embedding_size))  # (batch_size, seq_len, h)
        projected_outputs = self.combine_heads(concat_outputs)  # (batch_size, seq_len, h)
        return projected_outputs

    @property
    def output_shape(self):
        return torch.Size([self.embedding_size])


class TransformerBlock(LudwigModule):
    def __init__(
        self,
        input_size: int,
        max_sequence_length: int,
        hidden_size: int,
        num_heads: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size

        self.self_attention = MultiHeadSelfAttention(input_size, hidden_size, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.fully_connected = nn.Sequential(
            nn.Linear(input_size, output_size), get_activation("relu"), nn.Linear(output_size, hidden_size)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length, self.input_size])

    def forward(self, inputs, mask=None):
        # inputs [b, s, h]
        attn_output = self.self_attention(inputs)  # [b, s, h]
        attn_output = self.dropout1(attn_output)  # [b, s, h]
        ln1_output = self.layernorm1(inputs + attn_output)  # [b, s, h]
        fc_output = self.fully_connected(ln1_output)  # [b, s, h]
        fc_output = self.dropout2(fc_output)  # [b, s, h]
        return self.layernorm2(ln1_output + fc_output)  # [b, s, h]

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length, self.hidden_size])


class TransformerStack(LudwigModule):
    def __init__(
        self,
        input_size: int,
        max_sequence_length: int,
        hidden_size: int = 256,
        num_heads: int = 8,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.supports_masking = True
        self.max_sequence_length = max_sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()

        prior_input_size = input_size
        for i in range(num_layers):
            layer = TransformerBlock(
                input_size=prior_input_size,
                max_sequence_length=max_sequence_length,
                hidden_size=hidden_size,
                num_heads=num_heads,
                output_size=output_size,
                dropout=dropout,
            )
            self.layers.append(layer)
            prior_input_size = self.layers[i].output_shape[-1]

        for layer in self.layers:
            logger.debug(f"   {layer._get_name()}")

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length, self.input_size])

    def forward(self, inputs, mask=None):
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, mask=mask)
        return hidden

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length, self.hidden_size])


# todo future: maybe reintroduce these attention function
# def feed_forward_attention(current_inputs, feature_hidden_size,
#                            hidden_size=256):
#     with tf.variable_scope('ff_attention'):
#         geated_inputs = reduce_feed_forward_attention(current_inputs,
#                                                       hidden_size=hidden_size)
#
#         # stacking inputs and attention vectors
#         tiled_geated_inputs = tf.tile(tf.expand_dims(geated_inputs, 1),
#                                       [1, tf.shape(current_inputs)[1], 1])
#         logger.debug(
#             '  att_tiled_geated_inputs: {}'.format(tiled_geated_inputs))
#         outputs = tf.concat([current_inputs, tiled_geated_inputs],
#                             axis=-1)  # [bs x s1 x 2*h]
#         logger.debug('  att_outputs: {}'.format(outputs))
#         # outputs = current_inputs + context # [bs x s1 x h]
#
#     return outputs, feature_hidden_size * 2
#
#
# todo future: maybe reintroduce these attention function
# def simple_memory_attention(current_inputs, context):
#     assert current_inputs.shape[2] == context.shape[2]
#     # calculating attention
#     attention = tf.nn.softmax(
#         tf.matmul(current_inputs, context, transpose_b=True))  # [bs x s1 x s2]
#     logger.debug('  att_outputs: {}'.format(attention))
#
#     # weighted_sum(attention, encoding_sequence_embedded)
#     exp_ese = tf.expand_dims(context, 1)  # [bs x 1 x s2 x h]
#     exp_att = tf.expand_dims(attention, -1)  # [bs x s1 x s2 x 1]
#     weighted_sum = tf.multiply(exp_ese, exp_att)  # [bs x s1 x s2 x h]
#     reduced_weighted_sum = tf.reduce_sum(weighted_sum, axis=2)  # [bs x s1 x h]
#     logger.debug('  att_reduced_weighted_sum: {}'.format(reduced_weighted_sum))
#
#     # stacking inputs and attention vectors
#     outputs = tf.concat([current_inputs, reduced_weighted_sum],
#                         axis=-1)  # [bs x s1 x 2*h]
#     logger.debug('  att_outputs: {}'.format(outputs))
#
#     return outputs, outputs.shape[-1]
#
#
# todo future: maybe reintroduce these attention function
# def feed_forward_memory_attention(current_inputs, memory, hidden_size=256):
#     seq_len = tf.shape(current_inputs)[1]
#     mem_len = tf.shape(current_inputs)[1]
#     seq_width = current_inputs.shape[2]
#     mem_width = memory.shape[2]
#
#     inputs_tile = tf.reshape(tf.tile(current_inputs, [1, 1, mem_len]),
#                              [-1, seq_len, mem_len, seq_width])
#     context_tile = tf.reshape(tf.tile(memory, [1, seq_len, 1]),
#                               [-1, seq_len, mem_len, mem_width])
#     concat_tile = tf.concat([inputs_tile, context_tile],
#                             axis=-1)  # [bs, seq, seq, seq_w + ctx_w]
#     logger.debug('  att_input_context_concat: {}'.format(concat_tile))
#
#     with tf.variable_scope('reduce_contextual_ff_attention'):
#         weights_1 = tf.get_variable('weights_1',
#                                     [concat_tile.shape[-1], hidden_size])
#         logger.debug('  att_weights_1: {}'.format(weights_1))
#         biases_1 = tf.get_variable('biases_1', [hidden_size])
#         logger.debug('  att_biases_1: {}'.format(biases_1))
#         weights_2 = tf.get_variable('weights_2', [hidden_size, 1])
#         logger.debug('  att_weights_2: {}'.format(weights_2))
#
#         current_inputs_reshape = tf.reshape(concat_tile,
#                                             [-1, concat_tile.shape[-1]])
#         hidden = tf.tanh(
#             tf.matmul(current_inputs_reshape, weights_1) + biases_1)
#         logger.debug('  att_hidden: {}'.format(hidden))
#         attention = tf.nn.softmax(
#             tf.reshape(tf.matmul(hidden, weights_2), [-1, seq_len, mem_len]))
#         logger.debug('  att_attention: {}'.format(attention))
#         # attention [bs x seq]
#         geated_inputs = tf.reduce_sum(
#             tf.expand_dims(attention, -1) * inputs_tile, 2)
#         logger.debug('  att_geated_inputs: {}'.format(geated_inputs))
#
#     return geated_inputs, geated_inputs.shape.as_list()[-1]
