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

from collections import UserDict
from typing import Generic, TypeVar

DEFAULT_KEYS = ["None", "none", "null", None]


T = TypeVar("T")


class Registry(UserDict, Generic[T]):
    """Registry is like a normal dict, but with an optional parent dict.

    Items are considered to exist in the registry if they are added to either the registry itself, or its parent.
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

    def __getitem__(self, key: str) -> T:
        if self.parent and key not in self.data:
            return self.parent.__getitem__(key)
        return self.data.__getitem__(key)

    def __contains__(self, key: str):
        return key in self.data or key in self.parent

    def __len__(self) -> int:
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
        return {**self.parent, **self.data}

    def register(self, name: str, default: bool = False):
        def wrap(cls):
            self[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    self[key] = cls
            return cls

        return wrap
