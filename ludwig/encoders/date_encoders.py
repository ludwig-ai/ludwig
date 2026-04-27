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
from abc import abstractmethod

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DATE, DATE_VECTOR_LENGTH, ENCODER_OUTPUT
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.date_encoders import DateEmbedConfig, DateWaveConfig
from ludwig.utils import torch_utils

logger = logging.getLogger(__name__)

# Date components and their properties.
# (name, vocabulary_size, periodic_period, index, needs_offset)
# vocabulary_size is used by DateEmbed for categorical embeddings.
# periodic_period is used by DateWave for sinusoidal encoding.
# needs_offset: whether to subtract 1 from the input (1-indexed -> 0-indexed).
DATE_COMPONENTS = [
    # name,        vocab, period, col, offset
    ("month", 12, 12, 1, True),
    ("day", 31, 31, 2, True),
    ("weekday", 7, 7, 3, False),
    ("yearday", 366, 366, 4, True),
    ("hour", 24, 24, 5, False),
    ("minute", 60, 60, 6, False),
    ("second", 60, 60, 7, False),
]

# second_of_day is always encoded with periodic encoding in both DateEmbed and DateWave.
SECOND_OF_DAY_PERIOD = 86400


@DeveloperAPI
class DateEncoderBase(Encoder):
    """Base class for date encoders providing shared infrastructure.

    Date features are preprocessed into a fixed-size vector of 9 integer components:
    [year, month, day, weekday, yearday, hour, minute, second, second_of_day].

    Subclasses must implement ``encode_components()`` to define how individual
    date components are encoded (e.g., categorical embeddings vs. sinusoidal
    periodic encoding). The base class handles year scaling via a learned FC layer,
    second_of_day periodic encoding, concatenation, and the final FC stack.
    """

    def __init__(
        self,
        fc_layers: list[dict] | None = None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: str | None = None,
        norm_params: dict | None = None,
        activation: str = "relu",
        dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        # Year is always encoded with a learned linear projection (continuous value).
        logger.debug("  year FCStack")
        self.year_fc = FCStack(
            first_layer_input_size=1,
            num_layers=1,
            default_output_size=1,
            default_use_bias=use_bias,
            default_weights_initializer=weights_initializer,
            default_bias_initializer=bias_initializer,
            default_norm=None,
            default_norm_params=None,
            default_activation=None,
            default_dropout=dropout,
        )

        # Store FC stack params for use after subclass sets up component encoders.
        self._fc_stack_params = dict(
            fc_layers=fc_layers,
            num_fc_layers=num_fc_layers,
            output_size=output_size,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            norm=norm,
            norm_params=norm_params,
            activation=activation,
            dropout=dropout,
        )

    def _build_fc_stack(self, component_output_size: int):
        """Build the final FC stack given the total size of encoded components.

        Args:
            component_output_size: Sum of output sizes from all component encoders
                (excluding year_fc and second_of_day periodic, which are added here).
        """
        # year_fc output (1) + component encodings + second_of_day periodic (1)
        fc_layer_input_size = self.year_fc.output_shape[0] + component_output_size + 1

        p = self._fc_stack_params
        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=fc_layer_input_size,
            layers=p["fc_layers"],
            num_layers=p["num_fc_layers"],
            default_output_size=p["output_size"],
            default_use_bias=p["use_bias"],
            default_weights_initializer=p["weights_initializer"],
            default_bias_initializer=p["bias_initializer"],
            default_norm=p["norm"],
            default_norm_params=p["norm_params"],
            default_activation=p["activation"],
            default_dropout=p["dropout"],
        )
        # Clean up stored params after use.
        del self._fc_stack_params

    @abstractmethod
    def encode_components(self, input_vector: torch.Tensor) -> list[torch.Tensor]:
        """Encode the date components (month through second) into a list of tensors.

        Args:
            input_vector: Integer tensor of shape [batch, DATE_VECTOR_LENGTH].

        Returns:
            List of tensors, one per component, each of shape [batch, encoding_dim].
        """
        ...

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        """Encode a date feature vector.

        Args:
            inputs: Tensor of shape [batch, DATE_VECTOR_LENGTH] with dtype int or float,
                containing [year, month, day, weekday, yearday, hour, minute, second, second_of_day].

        Returns:
            Dictionary with ENCODER_OUTPUT key mapping to tensor of shape [batch, output_size].
        """
        input_vector = inputs.to(torch.int)

        # Year: continuous, passed through a learned linear layer.
        scaled_year = self.year_fc(input_vector[:, 0:1].to(torch.float))

        # Components (month..second): encoded by subclass.
        encoded_components = self.encode_components(input_vector)

        # second_of_day: always periodic (shared by both embed and wave encoders).
        periodic_second_of_day = torch_utils.periodic(input_vector[:, 8:9].to(torch.float), SECOND_OF_DAY_PERIOD)

        hidden = torch.cat([scaled_year] + encoded_components + [periodic_second_of_day], dim=1)

        hidden = self.fc_stack(hidden)
        return {ENCODER_OUTPUT: hidden}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([DATE_VECTOR_LENGTH])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@DeveloperAPI
@register_encoder("embed", DATE)
class DateEmbed(DateEncoderBase):
    """Encodes date components using learned categorical embeddings.

    Each cyclic date component (month, day, weekday, yearday, hour, minute, second) is mapped to a dense embedding
    vector via a lookup table, similar to how categorical features are encoded. The year is projected through a linear
    layer, and second_of_day uses a periodic (cosine) encoding. All representations are concatenated and passed through
    an optional FC stack.

    Use this encoder when you want the model to learn arbitrary (non-sinusoidal) representations for each date
    component. This is the default date encoder and works well in most scenarios. For datasets where the cyclic nature
    of time components is important (e.g., hourly patterns wrapping around midnight), consider ``DateWave``.
    """

    def __init__(
        self,
        embedding_size: int = 10,
        embeddings_on_cpu: bool = False,
        fc_layers: list[dict] | None = None,
        num_fc_layers: int = 0,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: str | None = None,
        norm_params: dict | None = None,
        activation: str = "relu",
        dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__(
            fc_layers=fc_layers,
            num_fc_layers=num_fc_layers,
            output_size=output_size,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            norm=norm,
            norm_params=norm_params,
            activation=activation,
            dropout=dropout,
            encoder_config=encoder_config,
            **kwargs,
        )

        self.embedding_size = embedding_size

        # Build one embedding module per date component.
        self.embed_modules = torch.nn.ModuleDict()
        total_embed_size = 0
        for name, vocab_size, _period, _col, _offset in DATE_COMPONENTS:
            logger.debug(f"  {name} Embed")
            embed = Embed(
                [str(i) for i in range(vocab_size)],
                embedding_size,
                representation="dense",
                embeddings_trainable=True,
                pretrained_embeddings=None,
                embeddings_on_cpu=embeddings_on_cpu,
                dropout=dropout,
                embedding_initializer=weights_initializer,
            )
            self.embed_modules[name] = embed
            total_embed_size += embed.output_shape[0]

        self._build_fc_stack(total_embed_size)

    def encode_components(self, input_vector: torch.Tensor) -> list[torch.Tensor]:
        """Encode date components using categorical embeddings."""
        encoded = []
        for name, _vocab, _period, col, offset in DATE_COMPONENTS:
            val = input_vector[:, col : col + 1]
            if offset:
                val = val - 1
            encoded.append(self.embed_modules[name](val))
        return encoded

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return DateEmbedConfig


@DeveloperAPI
@register_encoder("wave", DATE)
class DateWave(DateEncoderBase):
    """Encodes date components using periodic sinusoidal (cosine) functions.

    Each cyclic date component is encoded as cos(2*pi*x/period), which naturally captures the cyclic nature of time --
    e.g., hour 23 is close to hour 0, and December is close to January. The year is projected through a linear layer.
    All representations are concatenated and passed through an FC stack.

    This encoding is parameter-free for the components (no learned embeddings), making it more compact than
    ``DateEmbed``. It is inspired by the positional encoding approach from Vaswani et al., "Attention Is All You Need"
    (2017).

    Use this encoder when cyclic continuity matters and you want a lightweight encoding. For richer learned
    representations, use ``DateEmbed``.
    """

    def __init__(
        self,
        fc_layers: list[FCStack] | None = None,
        num_fc_layers: int = 1,
        output_size: int = 10,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        norm: str | None = None,
        norm_params: dict | None = None,
        activation: str = "relu",
        dropout: float = 0,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__(
            fc_layers=fc_layers,
            num_fc_layers=num_fc_layers,
            output_size=output_size,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            norm=norm,
            norm_params=norm_params,
            activation=activation,
            dropout=dropout,
            encoder_config=encoder_config,
            **kwargs,
        )

        # Each periodic component produces 1 output (cosine value).
        # 7 components (month..second), each producing 1 value.
        total_component_size = len(DATE_COMPONENTS)
        self._build_fc_stack(total_component_size)

    def encode_components(self, input_vector: torch.Tensor) -> list[torch.Tensor]:
        """Encode date components using periodic cosine functions."""
        input_float = input_vector.to(torch.float)
        encoded = []
        for _name, _vocab, period, col, _offset in DATE_COMPONENTS:
            encoded.append(torch_utils.periodic(input_float[:, col : col + 1], period))
        return encoded

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return DateWaveConfig
