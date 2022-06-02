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
import numpy as np
import pandas as pd

from ludwig.data.dataframe.base import DataFrameEngine


class PandasEngine(DataFrameEngine):
    def __init__(self, **kwargs):
        super().__init__()

    def df_like(self, df, proc_cols):
        # df argument unused for pandas, which can instantiate df directly
        return pd.DataFrame(proc_cols)

    def parallelize(self, data):
        return data

    def persist(self, data):
        return data

    def compute(self, data):
        return data

    def concat(self, dfs):
        return pd.concat(dfs)

    def from_pandas(self, df):
        return df

    def map_objects(self, series, map_fn, meta=None):
        return series.map(map_fn)

    def map_partitions(self, series, map_fn, meta=None):
        return map_fn(series)

    def apply_objects(self, df, apply_fn, meta=None):
        return df.apply(apply_fn, axis=1)

    def reduce_objects(self, series, reduce_fn):
        return reduce_fn(series)

    def to_parquet(self, df, path):
        df.to_parquet(path, engine="pyarrow")

    def to_ray_dataset(self, df):
        from ray.data import from_pandas

        return from_pandas(df)

    def from_ray_dataset(self, dataset) -> pd.DataFrame:
        return dataset.to_pandas()

    @property
    def array_lib(self):
        return np

    @property
    def df_lib(self):
        return pd

    @property
    def partitioned(self):
        return False

    def set_parallelism(self, parallelism):
        pass


PANDAS = PandasEngine()
