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
from functools import partial

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, LOSS, NUMBER, SEQUENCE, SET, TEXT, TYPE, VECTOR
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.schema.decoders.base import ClassifierConfig, PassthroughDecoderConfig, ProjectorConfig, RegressorConfig
from ludwig.utils.torch_utils import Dense, get_activation

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_decoder("passthrough", [BINARY, CATEGORY, NUMBER, SET, VECTOR, SEQUENCE, TEXT])
class PassthroughDecoder(Decoder):
    def __init__(self, input_size: int = 1, num_classes: int = None, decoder_config=None, **kwargs):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, inputs, **kwargs):
        return inputs

    @staticmethod
    def get_schema_cls():
        return PassthroughDecoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape


@DeveloperAPI
@register_decoder("regressor", [BINARY, NUMBER])
class Regressor(Decoder):
    def __init__(
        self,
        input_size,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Dense")

        self.dense = Dense(
            input_size=input_size,
            output_size=1,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

    @staticmethod
    def get_schema_cls():
        return RegressorConfig

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        return self.dense(inputs)


@DeveloperAPI
@register_decoder("projector", [VECTOR])
class Projector(Decoder):
    def __init__(
        self,
        input_size,
        output_size,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        activation=None,
        clip=None,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Dense")
        self.dense = Dense(
            input_size=input_size,
            output_size=output_size,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

        self.activation = get_activation(activation)

        if clip is not None:
            if isinstance(clip, (list, tuple)) and len(clip) == 2:
                self.clip = partial(torch.clip, min=clip[0], max=clip[1])
            else:
                raise ValueError(
                    "The clip parameter of {} is {}. "
                    "It must be a list or a tuple of length 2.".format(self.feature_name, self.clip)
                )
        else:
            self.clip = None

    @staticmethod
    def get_schema_cls():
        return ProjectorConfig

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        values = self.activation(self.dense(inputs))
        if self.clip:
            values = self.clip(values)
        return values


@DeveloperAPI
@register_decoder("classifier", [CATEGORY, SET])
class Classifier(Decoder):
    def __init__(
        self,
        input_size,
        num_classes,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Dense")
        self.num_classes = num_classes
        self.dense = Dense(
            input_size=input_size,
            output_size=num_classes,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

        self.sampled_loss = False
        if LOSS in kwargs and TYPE in kwargs[LOSS] and kwargs[LOSS][TYPE] is not None:
            self.sampled_loss = kwargs[LOSS][TYPE].startswith("sampled")

    @staticmethod
    def get_schema_cls():
        return ClassifierConfig

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        return self.dense(inputs)
