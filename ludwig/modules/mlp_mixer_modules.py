# Copyright (c) 2021 Linux Foundation
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
from typing import Tuple, Union

import torch
import torch.nn as nn

from ludwig.utils.torch_utils import LudwigModule


class MLP(LudwigModule):
    def __init__(
        self,
        in_features: Union[int, Tuple[int]],
        hidden_size: int,
        out_features: Union[int, Tuple[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        out_features = out_features or in_features

        self._input_shape = in_features
        self._output_shape = out_features

        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=out_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, **kwargs):
        hidden = self.dropout1(nn.functional.gelu(self.linear1(inputs)))
        return self.dropout2(self.linear2(hidden))

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self._input_shape])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_shape])


class MixerBlock(LudwigModule):
    def __init__(self, embed_size: int, n_patches: int, token_dim: int, channel_dim: int, dropout: float = 0.0):
        super().__init__()
        self._input_shape = (n_patches, embed_size)
        self._output_shape = (n_patches, embed_size)

        self.mlp1 = MLP(in_features=n_patches, hidden_size=token_dim, dropout=dropout)

        self.mlp2 = MLP(in_features=embed_size, hidden_size=channel_dim, dropout=dropout)

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_size)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_size)

    def forward(self, inputs: torch.Tensor, **kwargs):
        assert inputs.shape[1:] == self.input_shape

        hidden = inputs
        hidden = self.layernorm1(hidden).transpose(1, 2)
        hidden = self.mlp1(hidden).transpose(1, 2)

        mid = hidden + inputs

        hidden = self.layernorm2(mid)
        hidden = self.mlp2(hidden)

        output = hidden + mid
        assert output.shape[1:] == self.output_shape
        return output

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self._output_shape)


class MLPMixer(LudwigModule):
    """MLPMixer.

    Implements
    MLP-Mixer: An all-MLP Architecture for Vision
    https://arxiv.org/abs/2105.01601
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        patch_size: int = 16,
        embed_size: int = 512,
        token_size: int = 2048,
        channel_dim: int = 256,
        num_layers: int = 8,
        dropout: float = 0.0,
        avg_pool: bool = True,
    ):
        super().__init__()
        assert (img_height % patch_size == 0) and (img_width % patch_size == 0)

        self._input_shape = (in_channels, img_height, img_width)
        n_patches = int(img_height * img_width / (patch_size**2))

        self.patch_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=embed_size, kernel_size=patch_size, stride=patch_size
        )

        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(
                    embed_size=embed_size,
                    n_patches=n_patches,
                    token_dim=token_size,
                    channel_dim=channel_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=embed_size)

        self.avg_pool = avg_pool
        if self.avg_pool:
            self._output_shape = torch.Size((embed_size,))
        else:
            self._output_shape = torch.Size((n_patches, embed_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1:] == self.input_shape
        hidden = self.patch_conv(inputs)
        hidden = hidden.flatten(2).transpose(1, 2)

        for mixer_block in self.mixer_blocks:
            hidden = mixer_block(hidden)
        hidden = self.layer_norm(hidden)

        if self.avg_pool:
            hidden = torch.mean(hidden, dim=1)

        assert hidden.shape[1:] == self.output_shape

        return hidden

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    @property
    def output_shape(self) -> torch.Size:
        return self._output_shape
