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

import os
import tempfile
import uuid

from abc import ABC, abstractmethod
from contextlib import contextmanager

from ludwig.data.dataframe.pandas import PANDAS
from ludwig.models.predictor import Predictor
from ludwig.models.trainer import Trainer
from ludwig.utils.tf_utils import initialize_tensorflow


class CacheMixin:
    _cache_dir: str

    @property
    def cache_enabled(self):
        return self._cache_dir is not None

    def create_cache_entry(self):
        return os.path.join(self.cache_dir, str(uuid.uuid1()))

    @property
    def cache_dir(self):
        if not self._cache_dir:
            raise ValueError('Cache directory not available, try calling `with backend.create_cache_dir()`.')
        return self._cache_dir

    @contextmanager
    def create_cache_dir(self):
        prev_cache_dir = self._cache_dir
        try:
            if self._cache_dir:
                os.makedirs(self._cache_dir, exist_ok=True)
                yield self._cache_dir
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    self._cache_dir = tmpdir
                    yield tmpdir
        finally:
            self._cache_dir = prev_cache_dir


class Backend(CacheMixin, ABC):
    def __init__(self, cache_dir=None):
        self._cache_dir = cache_dir

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def initialize_tensorflow(self, *args, **kwargs):
        raise NotImplementedError()

    @contextmanager
    @abstractmethod
    def create_trainer(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def sync_model(self, model):
        raise NotImplementedError()

    @abstractmethod
    def broadcast_return(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def is_coordinator(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def df_engine(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def supports_multiprocessing(self):
        raise NotImplementedError()

    @abstractmethod
    def check_lazy_load_supported(self, feature):
        raise NotImplementedError()


class LocalPreprocessingMixin:
    @property
    def df_engine(self):
        return PANDAS

    @property
    def supports_multiprocessing(self):
        return True

    def check_lazy_load_supported(self, feature):
        pass


class LocalTrainingMixin:
    def initialize_tensorflow(self, *args, **kwargs):
        initialize_tensorflow(*args, **kwargs)

    def create_trainer(self, **kwargs):
        return Trainer(**kwargs)

    def create_predictor(self, **kwargs):
        return Predictor(**kwargs)

    def sync_model(self, model):
        pass

    def broadcast_return(self, fn):
        return fn()

    def is_coordinator(self):
        return True


class RemoteTrainingMixin:
    def sync_model(self, model):
        pass

    def broadcast_return(self, fn):
        return fn()

    def is_coordinator(self):
        return True


class LocalBackend(LocalPreprocessingMixin, LocalTrainingMixin, Backend):
    def __init__(self):
        super().__init__()

    def initialize(self):
        pass
