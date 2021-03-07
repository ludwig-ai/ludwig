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

from collections import UserDict

import ludwig.utils


DEFAULT_KEYS = ['None', 'none', 'null', None]


class Registry(UserDict):
    """Registry is like a normal dict, but with an optional parent dict.

    Items are considered to exist in the registry if they are added to either
    the registry itself, or its parent.
    """
    def __init__(self, source=None):
        init_data = None
        parent = {}
        if isinstance(source, Registry):
            parent = source
        else:
            init_data = source

        self.parent = parent
        super().__init__(init_data)

    def __getitem__(self, key):
        if self.parent and key not in self.data:
            return self.parent.__getitem__(key)
        return self.data.__getitem__(key)

    def __contains__(self, key):
        return key in self.data or key in self.parent

    def __len__(self):
        return len(self.data) + len(self.parent)

    def __iter__(self):
        return self._merged().__iter__()

    def keys(self):
        return self._merged().keys()

    def values(self):
        return self._merged().values()

    def items(self):
        return self._merged().items()

    def _merged(self):
        return {
            **self.parent,
            **self.data
        }


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
