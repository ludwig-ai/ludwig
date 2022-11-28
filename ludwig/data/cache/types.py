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

import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.fs_utils import checksum
from ludwig.utils.types import DataFrame


def alphanum(v):
    """Filters a string to only its alphanumeric characters."""
    return re.sub(r"\W+", "", v)


@DeveloperAPI
class CacheableDataset(ABC):
    name: str
    checksum: str

    @abstractmethod
    def get_cache_path(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_cache_directory(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def unwrap(self) -> Union[str, DataFrame]:
        raise NotImplementedError()


@DeveloperAPI
@dataclass
class CacheableDataframe(CacheableDataset):
    df: DataFrame
    name: str
    checksum: str

    def get_cache_path(self) -> str:
        return alphanum(self.name)

    def get_cache_directory(self) -> str:
        return os.getcwd()

    def unwrap(self) -> Union[str, DataFrame]:
        return self.df


@DeveloperAPI
@dataclass
class CacheablePath(CacheableDataset):
    path: str

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def checksum(self) -> str:
        return checksum(self.path)

    def get_cache_path(self) -> str:
        return self.name

    def get_cache_directory(self) -> str:
        return os.path.dirname(self.path)

    def unwrap(self) -> Union[str, DataFrame]:
        return self.path


CacheInput = Union[str, DataFrame, CacheableDataset]


def wrap(dataset: Optional[CacheInput]) -> CacheableDataset:
    if dataset is None:
        return None

    if isinstance(dataset, CacheableDataset):
        return dataset
    if isinstance(dataset, str):
        return CacheablePath(path=dataset)

    # TODO(travis): could try hashing the in-memory dataset, but this is tricky for Dask
    checksum = str(uuid.uuid1())
    name = checksum
    return CacheableDataframe(df=dataset, name=name, checksum=checksum)
