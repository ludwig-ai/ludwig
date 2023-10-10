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
import os

import numpy as np
import pandas as pd

from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.globals import PREDICTIONS_SHAPES_FILE_NAME
from ludwig.utils.data_utils import load_json, save_json, split_by_slices
from ludwig.utils.dataframe_utils import flatten_df, unflatten_df


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

    @staticmethod
    def concat(dfs) -> pd.DataFrame:
        return pd.concat(dfs)

    def from_pandas(self, df):
        return df

    def map_objects(self, series, map_fn, meta=None):
        return series.map(map_fn)

    def map_batches(self, df, map_fn, enable_tensor_extension_casting=True):
        return map_fn(df)

    def map_partitions(self, series, map_fn, meta=None):
        return map_fn(series)

    def apply_objects(self, df, apply_fn, meta=None):
        return df.apply(apply_fn, axis=1)

    def reduce_objects(self, series, reduce_fn):
        return reduce_fn(series)

    def split(self, df, probabilities):
        return split_by_slices(df.iloc, len(df), probabilities)

    @staticmethod
    def remove_empty_partitions(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def to_parquet(self, df, path, index=False):
        df.to_parquet(path, engine="pyarrow", index=index)

    def write_predictions(self, df: pd.DataFrame, path: str):
        df, column_shapes = flatten_df(df, self)
        self.to_parquet(df, path)
        save_json(os.path.join(os.path.dirname(path), PREDICTIONS_SHAPES_FILE_NAME), column_shapes)

    def read_predictions(self, path: str) -> pd.DataFrame:
        pred_df = pd.read_parquet(path)
        column_shapes = load_json(os.path.join(os.path.dirname(path), PREDICTIONS_SHAPES_FILE_NAME))
        return unflatten_df(pred_df, column_shapes, self)

    def to_ray_dataset(self, df):
        from ray.data import from_pandas

        return from_pandas(df)

    @staticmethod
    def from_ray_dataset(dataset) -> pd.DataFrame:
        return dataset.to_pandas()

    @staticmethod
    def reset_index(df) -> pd.DataFrame:
        return df.reset_index(drop=True)

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
