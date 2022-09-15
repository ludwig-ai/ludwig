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
from typing import Any, Dict

import torch

from ludwig.modules.ludwig_module import LudwigModule, LudwigModuleState
from ludwig.schema.encoders.base import BaseEncoderConfig


class Encoder(LudwigModule, ABC):
    def __init__(self, config: BaseEncoderConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def restore_from_state(cls, state: LudwigModuleState) -> "Encoder":
        schema = cls.get_schema_cls().Schema()
        encoder_config = schema.load(state.config)
        encoder = cls(encoder_config)
        encoder.load_state_dict({k: torch.from_numpy(v) for k, v in state.saved_weights.items()})
        return encoder

    def get_state(self, metadata: Dict[str, Any] = None) -> LudwigModuleState:
        return super().get_state(
            config=self.config.Schema().dump(self.config),
            metadata=metadata,
        )
