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

from ludwig.data.engine.dask import DaskEngine
from ludwig.data.engine.pandas import PandasEngine
from ludwig.utils.misc_utils import get_from_registry


ENGINES = [PandasEngine(), DaskEngine()]

processing_engine_registry = {
    dtype: engine for engine in ENGINES for dtype in engine.dtypes
}


def get_processing_engine(df):
    return get_from_registry(
        type(df),
        processing_engine_registry
    )
