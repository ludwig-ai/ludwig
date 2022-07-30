#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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
import copy
from collections import defaultdict
from functools import total_ordering
from typing import Callable, Dict, List, Optional


@total_ordering
class VersionTransformation:
    def __init__(self, transform: Callable[[Dict], Dict], version: str, prefixes: List[str] = None):
        self.transform = transform
        self.version = version
        self.prefixes = prefixes if prefixes else []

    def transform_config(self, config: Dict):
        for prefix in self.prefixes:
            config = self.transform_config_with_prefix(config, prefix)
        return config

    def transform_config_with_prefix(self, config: Dict, prefix: Optional[str] = None) -> Dict:
        """Applied this version transformation to the config, returns the updated config."""
        if prefix:
            components = prefix.split(".", 1)
            key = components[0]
            rest_of_prefix = key[1] if len(key) > 1 else ""
            if key in config:
                subsection = config[key]
                if isinstance(subsection, list):
                    config[key] = [self.transform_config(v) if isinstance(v, dict) else v for v in subsection]
                elif isinstance(subsection, dict):
                    config[key] = self.transform_config(subsection, prefix=rest_of_prefix)
            return config
        else:
            # Base case: no prefix specified, pass entire dictionary to transform function.
            return self.transform(config)

    @property
    def prefix_length(self):
        return len(self.prefix.split(".")) if self.prefix else 0

    def __lt__(self, other):
        """Defines sort order of version transformations. Sorted by:

        - version (ascending)
        - prefix_length (ascending)  Process outer config transformations before inner.
        - prefix (ascending)
        """
        return (self.version, self.prefix_length, self.prefix) < (other.version, other.prefix_length, other.prefix)

    def __repr__(self):
        return f'VersionTransformation(<function>, version="{self.version}", prefix="{self.prefix}")'


class VersionTransformationRegistry:
    """Allows callers to register transformations which update versioned config files."""

    def __init__(self):
        self._registry = defaultdict(list)  # Maps version number to list of transformations.

    def register(self, transformation: VersionTransformation):
        self._registry[transformation.version].append(transformation)

    def get_transformations(self, from_version: str, to_version: str) -> List[VersionTransformation]:
        """Get the config transformations from one version to the next."""
        versions = [v for v in self._registry.keys() if v <= to_version and v > from_version]
        transforms = sorted(t for v in versions for t in self._registry[v])
        return transforms

    def update_config(self, config: Dict, from_version: str, to_version: str) -> Dict:
        """Applies the transformations from an older version to a newer version."""
        transformations = self.get_transformations(from_version, to_version)
        updated_config = copy.deepcopy(config)
        for t in transformations:
            updated_config = t.transform_config(updated_config)
        updated_config["ludwig_version"] = to_version
        return updated_config
