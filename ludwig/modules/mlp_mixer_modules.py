# coding=utf-8
# Copyright (c) 2021 Linuf Foundation
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
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, \
    Conv2D, GlobalAveragePooling1D

from ludwig.modules.activation_modules import gelu


class MLP(Layer):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

    def build(self, shape):
        self.dense1 = Dense(self.hidden_size)
        self.dense2 = Dense(shape[-1])
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)

    def call(self, inputs, **kwargs):
        hidden = self.dropout1(gelu(self.dense1(inputs)))
        return self.dropout2(self.dense2(hidden))


class MixerBlock(Layer):
    def __init__(self, token_dim, channel_dim, dropout=0.0):
        super().__init__()
        self.mlp1 = MLP(token_dim, dropout)
        self.mlp2 = MLP(channel_dim, dropout)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def call(self, inputs, **kwargs):
        hidden = inputs
        hidden = self.layernorm1(hidden)
        hidden = tf.transpose(hidden, [0, 2, 1])
        hidden = self.mlp1(hidden)
        hidden = tf.transpose(hidden, [0, 2, 1])
        mid = hidden + inputs
        hidden = self.layernorm2(mid)
        hidden = self.mlp2(hidden)
        return hidden + mid


class MLPMixer(Layer):
    """MLPMixer

    Implements
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            patch_size=16,
            embed_size=512,
            token_size=2048,
            channel_dim=256,
            num_layers=8,
            dropout=0.0,
            avg_pool=True,
    ):
        super().__init__()
        self.patch_conv = Conv2D(embed_size, patch_size, patch_size)
        self.mixer_blocks = [
            MixerBlock(token_size, channel_dim, dropout)
            for _ in range(num_layers)
        ]
        self.layer_norm = LayerNormalization()

        self.gap = None
        if avg_pool:
            self.avg_pool = GlobalAveragePooling1D()

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        hidden = inputs
        hidden = self.patch_conv(hidden)
        hidden = tf.reshape(hidden, [batch_size, -1, hidden.shape[-1]])
        for mixer_block in self.mixer_blocks:
            hidden = mixer_block(hidden)
        hidden = self.layer_norm(hidden)
        if self.avg_pool:
            hidden = self.avg_pool(hidden)
        return hidden
