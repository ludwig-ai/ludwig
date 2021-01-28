#! /usr/bin/env python
# coding=utf-8
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

from tensorflow.keras.layers import Layer

from ludwig.utils.registry import DEFAULT_KEYS


class Decoder(Layer, ABC):
    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def register(cls, name):
        raise NotImplementedError

    @classmethod
    def register_default(cls):
        for key in DEFAULT_KEYS:
            cls.register(name=key)
