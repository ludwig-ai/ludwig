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

from abc import ABC, abstractmethod

from ludwig.backend.context import Context
from ludwig.data.engine.pandas import PandasEngine


class Backend(Context, ABC):
    def __init__(self):
        super().__init__(Backend)

    @property
    @abstractmethod
    def processor(self):
        raise NotImplementedError()


class CompositeBackend(Backend):
    def __init__(self, processor):
        super().__init__()
        self._processor = processor

    @property
    def processor(self):
        return self._processor


class LocalBackend(Backend):
    def __init__(self):
        super().__init__()
        self._processor = PandasEngine()

    @property
    def processor(self):
        return self._processor
