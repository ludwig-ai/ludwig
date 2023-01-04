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
from typing import Any, Dict, Set

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import MODEL_ECD
from ludwig.utils.torch_utils import LudwigModule


@DeveloperAPI
class Encoder(LudwigModule, ABC):
    @abstractmethod
    def forward(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def get_fixed_preprocessing_params(cls, encoder_params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a dict of fixed preprocessing parameters for the encoder if required."""
        return {}

    @classmethod
    def is_pretrained(cls, encoder_params: Dict[str, Any]) -> bool:
        return False

    @classmethod
    def get_supported_model_types(cls, encoder_params: Dict[str, Any]) -> Set[str]:
        return {MODEL_ECD}
