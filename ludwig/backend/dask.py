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

from ludwig.backend.base import Backend, LocalTrainingMixin
from ludwig.constants import NAME
from ludwig.data.dataframe.dask import DaskEngine


class DaskBackend(LocalTrainingMixin, Backend):
    def __init__(self):
        super().__init__()
        self._df_engine = DaskEngine()

    def initialize(self):
        pass

    @property
    def df_engine(self):
        return self._df_engine

    @property
    def supports_multiprocessing(self):
        return False

    def check_lazy_load_supported(self, feature):
        raise ValueError(f'DaskBackend does not support lazy loading of data files at train time. '
                         f'Set preprocessing config `in_memory: True` for feature {feature[NAME]}')
