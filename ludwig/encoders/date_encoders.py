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
from typing import Dict

import torch

from ludwig.constants import DATE
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.embedding_modules import Embed
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.schema.encoders.date_encoders import DateEmbedConfig, DateWaveConfig
from ludwig.utils import torch_utils

logger = logging.getLogger(__name__)

# Year, month, day, weekday, yearday, hour, minute, seconds, second_of_day.
# TODO: Share this constant with date_feature.DATE_VECTOR_SIZE.
DATE_INPUT_SIZE = 9


@register_encoder("embed", DATE)
class DateEmbed(Encoder):
    def __init__(self, encoder_config: DateEmbedConfig = DateEmbedConfig()):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        logger.debug("  year FCStack")
        self.year_fc = FCStack(
            first_layer_input_size=1,
            num_layers=1,
            default_output_size=1,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=None,
            default_norm_params=None,
            default_activation=None,
            default_dropout=encoder_config.dropout,
        )

        logger.debug("  month Embed")
        self.embed_month = Embed(
            [str(i) for i in range(12)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  day Embed")
        self.embed_day = Embed(
            [str(i) for i in range(31)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  weekday Embed")
        self.embed_weekday = Embed(
            [str(i) for i in range(7)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  yearday Embed")
        self.embed_yearday = Embed(
            [str(i) for i in range(366)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  hour Embed")
        self.embed_hour = Embed(
            [str(i) for i in range(24)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  minute Embed")
        self.embed_minute = Embed(
            [str(i) for i in range(60)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        logger.debug("  second Embed")
        self.embed_second = Embed(
            [str(i) for i in range(60)],
            encoder_config.embedding_size,
            representation="dense",
            embeddings_trainable=True,
            pretrained_embeddings=None,
            embeddings_on_cpu=encoder_config.embeddings_on_cpu,
            dropout=encoder_config.dropout,
            embedding_initializer=encoder_config.weights_initializer,
        )

        # Summed sizes of all of the embeddings.
        fc_layer_input_size = (
            self.year_fc.output_shape[0]
            + self.embed_month.output_shape[0]
            + self.embed_day.output_shape[0]
            + self.embed_weekday.output_shape[0]
            + self.embed_yearday.output_shape[0]
            + self.embed_hour.output_shape[0]
            + self.embed_minute.output_shape[0]
            + self.embed_second.output_shape[0]
            + 1  # for periodic_second_of_day.
        )

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=fc_layer_input_size,
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

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The input vector fed into the encoder.
            Shape: [batch x DATE_INPUT_SIZE], type torch.int8
        :type inputs: Tensor
        """
        # ================ Embeddings ================
        input_vector = inputs.type(torch.int)

        scaled_year = self.year_fc(input_vector[:, 0:1].type(torch.float))
        embedded_month = self.embed_month(input_vector[:, 1:2] - 1)
        embedded_day = self.embed_day(input_vector[:, 2:3] - 1)
        embedded_weekday = self.embed_weekday(input_vector[:, 3:4])
        embedded_yearday = self.embed_yearday(input_vector[:, 4:5] - 1)
        embedded_hour = self.embed_hour(input_vector[:, 5:6])
        embedded_minute = self.embed_minute(input_vector[:, 6:7])
        embedded_second = self.embed_second(input_vector[:, 7:8])
        periodic_second_of_day = torch_utils.periodic(input_vector[:, 8:9].type(torch.float), 86400)

        hidden = torch.cat(
            [
                scaled_year,
                embedded_month,
                embedded_day,
                embedded_weekday,
                embedded_yearday,
                embedded_hour,
                embedded_minute,
                embedded_second,
                periodic_second_of_day,
            ],
            dim=1,
        )

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))

        hidden = self.fc_stack(hidden)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return DateEmbedConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([DATE_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape


@register_encoder("wave", DATE)
class DateWave(Encoder):
    def __init__(self, encoder_config: DateWaveConfig):
        super().__init__(encoder_config)
        logger.debug(f" {self.name}")

        logger.debug("  year FCStack")
        self.year_fc = FCStack(
            first_layer_input_size=1,
            num_layers=1,
            default_output_size=1,
            default_use_bias=encoder_config.use_bias,
            default_weights_initializer=encoder_config.weights_initializer,
            default_bias_initializer=encoder_config.bias_initializer,
            default_norm=None,
            default_norm_params=None,
            default_activation=None,
            default_dropout=encoder_config.dropout,
        )

        # Summed sizes of all of the embeddings.
        # Additional 8 for periodic_[month, day, ..., second_of_day].
        fc_layer_input_size = self.year_fc.output_shape[0] + 8

        logger.debug("  FCStack")
        self.fc_stack = FCStack(
            first_layer_input_size=fc_layer_input_size,
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

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The input vector fed into the encoder.
            Shape: [batch x DATE_INPUT_SIZE], type torch.int8
        :type inputs: Tensor
        """
        # ================ Embeddings ================
        input_vector = inputs.type(torch.float)
        scaled_year = self.year_fc(input_vector[:, 0:1])
        periodic_month = torch_utils.periodic(input_vector[:, 1:2], 12)
        periodic_day = torch_utils.periodic(input_vector[:, 2:3], 31)
        periodic_weekday = torch_utils.periodic(input_vector[:, 3:4], 7)
        periodic_yearday = torch_utils.periodic(input_vector[:, 4:5], 366)
        periodic_hour = torch_utils.periodic(input_vector[:, 5:6], 24)
        periodic_minute = torch_utils.periodic(input_vector[:, 6:7], 60)
        periodic_second = torch_utils.periodic(input_vector[:, 7:8], 60)
        periodic_second_of_day = torch_utils.periodic(input_vector[:, 8:9], 86400)

        hidden = torch.cat(
            [
                scaled_year,
                periodic_month,
                periodic_day,
                periodic_weekday,
                periodic_yearday,
                periodic_hour,
                periodic_minute,
                periodic_second,
                periodic_second_of_day,
            ],
            dim=1,
        )

        # ================ FC Stack ================
        # logger.debug('  flatten hidden: {0}'.format(hidden))

        hidden = self.fc_stack(hidden)
        return {"encoder_output": hidden}

    @staticmethod
    def get_schema_cls():
        return DateWaveConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([DATE_INPUT_SIZE])

    @property
    def output_shape(self) -> torch.Size:
        return self.fc_stack.output_shape
