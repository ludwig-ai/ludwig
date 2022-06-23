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

import contextlib
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    @abstractmethod
    def initialize_batcher(self, batch_size=128, should_shuffle=True, seed=0, ignore_last=False, horovod=None):
        raise NotImplementedError()

    def to_df(self):
        raise NotImplementedError()


class DatasetManager(ABC):
    @abstractmethod
    def create(self, dataset, config, training_set_metadata):
        raise NotImplementedError()

    @abstractmethod
    def save(self, cache_path, dataset, config, training_set_metadata, tag):
        raise NotImplementedError()

    @abstractmethod
    def can_cache(self, skip_save_processed_input):
        raise NotImplementedError()

    @property
    @abstractmethod
    def data_format(self):
        raise NotImplementedError()
