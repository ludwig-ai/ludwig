#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2020 Uber Technologies, Inc.
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

# Legacy default keys for backward compatibility.
# New code should use the explicit `default` parameter instead.
DEFAULT_KEYS = ["None", "none", "null", None]


T = TypeVar("T")


class Registry(UserDict, Generic[T]):
    """Type-safe registry with optional parent delegation and mock support.

    Items are considered to exist in the registry if they are in either the
    registry itself or its parent. Supports:
    - Generic typing: Registry[EncoderType], Registry[CombinerType]
    - Parent delegation for hierarchical registries
    - register() decorator for clean registration
    - unregister() for testing and dynamic removal
    - Mock support via context manager
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
        """Register a class in the registry via decorator.

        Args:
            name: Registration key.
            default: If True, also register under None/"none"/"null" keys.
        """

        def wrap(cls):
            self[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    self[key] = cls
            return cls

        return wrap

    def unregister(self, name: str):
        """Remove a registered item. Useful for testing.

        Args:
            name: Key to remove.

        Raises:
            KeyError if name is not registered.
        """
        if name in self.data:
            del self.data[name]
        else:
            raise KeyError(f"'{name}' is not registered")

    def get_default(self) -> T | None:
        """Get the default registered item (registered with default=True)."""
        for key in DEFAULT_KEYS:
            if key in self.data:
                return self.data[key]
        return None

    def list_registered(self) -> list[str]:
        """List all registered names (excluding default key aliases)."""
        return [k for k in self._merged() if k not in DEFAULT_KEYS]
