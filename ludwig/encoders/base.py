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
from collections import UserDict

from tensorflow.keras.layers import Layer


DEFAULT_KEYS = ['None', 'none', 'null', None]


class Registry(UserDict):
    """Registry is like a normal dict, but with optional child dicts.

    When an item is added / removed from the parent, the children will also have
    that item added or removed.
    """
    def __init__(self, parent=None):
        self.children = []
        if isinstance(parent, Registry):
            parent.children.append(self)
        super().__init__(parent)

    def __setitem__(self, key, value):
        for child in self.children:
            child.__setitem__(key, value)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        for child in self.children:
            child.__delitem__(key)
        super().__delitem__(key)


def register(name):
    def wrap(cls):
        cls.register(name)
        return cls
    return wrap


def register_default(name):
    def wrap(cls):
        cls.register(name)
        cls.register_default()
        return cls
    return wrap


class Encoder(Layer, ABC):
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
