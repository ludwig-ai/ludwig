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

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Iterable

from ludwig.data.batcher.base import Batcher
from ludwig.distributed import DistributedStrategy
from ludwig.features.base_feature import BaseFeature
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.types import DataFrame


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @contextlib.contextmanager
    @abstractmethod
    def initialize_batcher(
        self,
        batch_size: int = 128,
        should_shuffle: bool = True,
        random_seed: int = default_random_seed,
        ignore_last: bool = False,
        distributed: DistributedStrategy = None,
    ) -> Batcher:
        raise NotImplementedError()

    @abstractmethod
    def to_df(self, features: Iterable[BaseFeature] | None = None) -> DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def to_scalar_df(self, features: Iterable[BaseFeature] | None = None) -> DataFrame:
        raise NotImplementedError()

    @property
    def in_memory_size_bytes(self) -> int:
        raise NotImplementedError()


class DatasetManager(ABC):
    @abstractmethod
    def create(self, dataset, config, training_set_metadata) -> Dataset:
        raise NotImplementedError()

    @abstractmethod
    def save(self, cache_path, dataset, config, training_set_metadata, tag) -> Dataset:
        raise NotImplementedError()

    @abstractmethod
    def can_cache(self, skip_save_processed_input) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def data_format(self) -> str:
        raise NotImplementedError()
