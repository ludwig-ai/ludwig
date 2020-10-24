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

import math

try:
    import dask
    import dask.array as da
    import dask.dataframe as dd
    import ray
    _LOADED = True
except:
    _LOADED = False

from ludwig.data.engine.base import DataProcessingEngine


def set_scheduler(scheduler):
    dask.config.set(scheduler=scheduler)


class DaskEngine(DataProcessingEngine):
    def parallelize(self, data):
        num_cpus = int(ray.cluster_resources().get('CPU', 1))
        return data.repartition(num_cpus)

    def persist(self, data):
        return data.persist()

    def compute(self, data):
        return data.compute()

    def array_to_col(self, array):
        return self.parallelize(dd.from_dask_array(array))

    @property
    def dtypes(self):
        if not _LOADED:
            return []
        return [dd.DataFrame]

    @property
    def array_lib(self):
        return da

    @property
    def df_lib(self):
        return dd
