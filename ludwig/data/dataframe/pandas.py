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
        # Our goal is to preserve the index of the input dataframe but to drop
        # all its columns. Because to_frame() creates a column from the index,
        # we need to drop it immediately following creation.
        col_names, cols = zip(*proc_cols.items())
        series_cols = []
        for col in cols:
            if type(col) not in {pd.Series, pd.DataFrame}:
                series_cols.append(pd.Series(col))
            else:
                series_cols.append(col)
        dataset = pd.concat(series_cols, join="inner", axis=1)  # inner join handles Series with dropped rows
        dataset.columns = col_names
        return dataset

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

    def from_ray_dataset(self, dataset):
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
