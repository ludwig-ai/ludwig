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

import multiprocessing
import os

import dask
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from ludwig.constants import NAME, PROC_COLUMN
from ludwig.data.dataset.parquet import ParquetDataset
from ludwig.data.dataset.partitioned import PartitionedDataset
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.utils.data_utils import DATA_PROCESSED_CACHE_DIR, DATASET_SPLIT_URL, DATA_TRAIN_HDF5_FP
from ludwig.utils.fs_utils import makedirs, to_url
from ludwig.utils.misc_utils import get_combined_features, get_proc_features

TMP_COLUMN = '__TMP_COLUMN__'


def set_scheduler(scheduler):
    dask.config.set(scheduler=scheduler)


class DaskEngine(DataFrameEngine):
    def __init__(self, parallelism=None, persist=False, **kwargs):
        self._parallelism = parallelism or multiprocessing.cpu_count()
        self._persist = persist

    def set_parallelism(self, parallelism):
        self._parallelism = parallelism

    def empty_df_like(self, df):
        # Our goal is to preserve the index of the input dataframe but to drop
        # all its columns. Because to_frame() creates a column from the index,
        # we need to drop it immediately following creation.
        return df.index.to_frame(name=TMP_COLUMN).drop(columns=[TMP_COLUMN])

    def parallelize(self, data):
        return data.repartition(self.parallelism)

    def persist(self, data):
        return data.persist() if self._persist else data

    def compute(self, data):
        return data.compute()

    def from_pandas(self, df):
        return dd.from_pandas(df, npartitions=self.parallelism)

    def map_objects(self, series, map_fn, meta=None):
        meta = meta or ('data', 'object')
        return series.map(map_fn, meta=meta)

    def apply_objects(self, df, apply_fn, meta=None):
        meta = meta or ('data', 'object')
        return df.apply(apply_fn, axis=1, meta=meta)

    def reduce_objects(self, series, reduce_fn):
        return series.reduction(reduce_fn, aggregate=reduce_fn, meta=('data', 'object')).compute()[0]

    def to_parquet(self, df, path):
        with ProgressBar():
            df.to_parquet(
                path,
                engine='pyarrow',
                write_index=False,
                schema='infer',
            )

    @property
    def array_lib(self):
        return da

    @property
    def df_lib(self):
        return dd

    @property
    def parallelism(self):
        return self._parallelism

    @property
    def partitioned(self):
        return True
