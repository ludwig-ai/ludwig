#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
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

from abc import ABC, abstractmethod

from torch import nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.torch_utils import LudwigModule


@DeveloperAPI
class Encoder(LudwigModule, ABC):
    @abstractmethod
    def forward(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def get_embedding_layer(self) -> nn.Module:
        """Returns layer that embeds inputs, used for computing explanations.

        Captum adds an evaluation hook to this module returned by this function. The hook copies the module's return
        with .clone(). The module returned by this function must return a tensor, not a dictionary of tensors.
        """
        return next(self.children())

    @property
    def name(self) -> str:
        return self.__class__.__name__
