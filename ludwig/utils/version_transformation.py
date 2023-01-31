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
import logging
from collections import defaultdict
from functools import total_ordering
from typing import Callable, Dict, List, Optional

from packaging import version as pkg_version

logger = logging.getLogger(__name__)


@total_ordering
class VersionTransformation:
    """Wrapper class for transformations to config dicts."""

    def __init__(self, transform: Callable[[Dict], Dict], version: str, prefixes: List[str] = None):
        """Constructor.

        Args:
            transform: A function or other callable from Dict -> Dict which returns a modified version of the config.
                       The callable may update the config in-place and return it, or return a new dict.
            version: The Ludwig version, should be the first version which requires this transform.
            prefixes: A list of config prefixes this transform should apply to, i.e. ["hyperopt"].  If not specified,
                      transform will be called with the entire config dictionary.
        """
        self.transform = transform
        self.version = version
        self.pkg_version = pkg_version.parse(version)
        self.prefixes = prefixes if prefixes else []

    def transform_config(self, config: Dict):
        """Transforms the sepcified config, returns the transformed config."""
        prefixes = self.prefixes if self.prefixes else [""]
        for prefix in prefixes:
            if prefix and (prefix not in config or not config[prefix]):
                # If the prefix is non-empty (transformation applies to a specific section), but the section is either
                # absent or empty, then skip.
                continue
            config = self.transform_config_with_prefix(config, prefix)
        return config

    def transform_config_with_prefix(self, config: Dict, prefix: Optional[str] = None) -> Dict:
        """Applied this version transformation to a specified prefix of the config, returns the updated config. If
        prefix names a list, i.e. "input_features", applies the transformation to each list element (input
        feature).

        Args:
            config: A config dictionary.
            prefix: An optional keypath prefix i.e. "input_features". If no prefix specified, transformation is applied
                    to config itself.

        Returns The updated config.
        """
        if prefix:
            components = prefix.split(".", 1)
            key = components[0]
            rest_of_prefix = components[1] if len(components) > 1 else ""
            if key in config:
                subsection = config[key]
                if isinstance(subsection, list):
                    config[key] = [
                        self.transform_config_with_prefix(v, prefix=rest_of_prefix) if isinstance(v, dict) else v
                        for v in subsection
                    ]
                elif isinstance(subsection, dict):
                    config[key] = self.transform_config_with_prefix(subsection, prefix=rest_of_prefix)
            return config
        else:
            # Base case: no prefix specified, pass entire dictionary to transform function.
            transformed_config = self.transform(config)
            if transformed_config is None:
                logger.error("Error: version transformation returned None. Check for missing return statement.")
            return transformed_config

    @property
    def max_prefix_length(self):
        """Returns the length of the longest prefix."""
        return max(len(prefix.split(".")) for prefix in self.prefixes) if self.prefixes else 0

    @property
    def longest_prefix(self):
        """Returns the longest prefix, or empty string if no prefixes specified."""
        prefixes = self.prefixes
        if not prefixes:
            return ""
        max_index = max(range(len(prefixes)), key=lambda i: prefixes[i])
        return prefixes[max_index]

    def __lt__(self, other):
        """Defines sort order of version transformations. Sorted by:

        - version (ascending)
        - max_prefix_length (ascending) Process outer config transformations before inner.
        - longest_prefix (ascending) Order alphabetically by prefix if max_prefix_length equal.
        """
        return (self.pkg_version, self.max_prefix_length, self.longest_prefix) < (
            other.pkg_version,
            other.max_prefix_length,
            other.longest_prefix,
        )

    def __repr__(self):
        return f'VersionTransformation(<function>, version="{self.version}", prefixes={repr(self.prefixes)})'


class VersionTransformationRegistry:
    """A registry of transformations which update versioned config files."""

    def __init__(self):
        self._registry = defaultdict(list)  # Maps version number to list of transformations.

    def register(self, transformation: VersionTransformation):
        """Registers a version transformation."""
        self._registry[transformation.version].append(transformation)

    def get_transformations(self, from_version: str, to_version: str) -> List[VersionTransformation]:
        """Filters transformations to create an ordered list of the config transformations from one version to
        another. All transformations returned have version st. from_version < version <= to_version.

        Args:
            from_version: The ludwig version of the input config.
            to_version: The version to update the config to (usually the current LUDWIG_VERSION).

        Returns an ordered list of transformations to apply to the config to update it.
        """
        from_version = pkg_version.parse(from_version)

        # Ignore pre-release, development versions. Otherwise transformations for upcoming releases will not be applied.
        to_version = pkg_version.parse(to_version)
        to_version = pkg_version.parse(f"{to_version.major}.{to_version.minor}")

        def in_range(v, to_version, from_version):
            v = pkg_version.parse(v)
            return from_version <= v <= to_version

        versions = [v for v in self._registry.keys() if in_range(v, to_version, from_version)]

        transforms = sorted(t for v in versions for t in self._registry[v])
        return transforms

    def update_config(self, config: Dict, from_version: str, to_version: str) -> Dict:
        """Applies the transformations from an older version to a newer version.

        Args:
            config: The config, created by ludwig at from_version.
            from_version: The version of ludwig which wrote the older config.
            to_version: The version of ludwig to update to (usually the current LUDWIG_VERSION).

        Returns The updated config after applying update transformations and updating the "ludwig_version" key.
        """
        transformations = self.get_transformations(from_version, to_version)
        updated_config = copy.deepcopy(config)
        for t in transformations:
            updated_config = t.transform_config(updated_config)
        updated_config["ludwig_version"] = to_version
        return updated_config
