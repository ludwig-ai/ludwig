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

from ludwig.constants import BINARY, CATEGORY, LOSS, NUMBER, SET, TYPE, VECTOR
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.utils.torch_utils import Dense, get_activation

logger = logging.getLogger(__name__)


@register_decoder("regressor", [BINARY, NUMBER], default=True)
class Regressor(Decoder):
    def __init__(
        self,
        input_size,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        activation=None,
        **kwargs,
    ):
        super().__init__()
        logger.debug(f" {self.name}")

        logger.debug("  Dense")

        self.dense = Dense(
            input_size=input_size,
            output_size=1,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        return self.dense(inputs)


@register_decoder("projector", [VECTOR], default=True)
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
        **kwargs,
    ):
        super().__init__()
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

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        values = self.activation(self.dense(inputs))
        if self.clip:
            values = self.clip(values)
        return values


@register_decoder("classifier", [CATEGORY, SET], default=True)
class Classifier(Decoder):
    def __init__(
        self,
        input_size,
        num_classes,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__()
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

        # this is needed because TF2 initializes the weights at the first call
        # so the first time we need to compute the full dense,
        # otherwise the weights of the Dense layer would not be initialized
        self.first_call = True

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        return self.dense(inputs)
