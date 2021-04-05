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

from ludwig.constants import NAME, PROC_COLUMN
from ludwig.data.dataset.parquet import ParquetDataset
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.utils.data_utils import DATA_PROCESSED_CACHE_DIR, DATASET_SPLIT_URL
from ludwig.utils.misc_utils import get_combined_features

TMP_COLUMN = '__TMP_COLUMN__'


def set_scheduler(scheduler):
    dask.config.set(scheduler=scheduler)


class DaskEngine(DataFrameEngine):
    def __init__(self):
        self._parallelism = multiprocessing.cpu_count()

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
        return data.persist()

    def compute(self, data):
        return data.compute()

    def from_pandas(self, df):
        return dd.from_pandas(df, npartitions=self.parallelism)

    def map_objects(self, series, map_fn):
        return series.map(map_fn, meta=('data', 'object'))

    def reduce_objects(self, series, reduce_fn):
        return series.reduction(reduce_fn, aggregate=reduce_fn, meta=('data', 'object')).compute()[0]

    def create_dataset(self, dataset, tag, config, training_set_metadata):
        cache_dir = training_set_metadata.get(DATA_PROCESSED_CACHE_DIR)
        tag = tag.lower()
        dataset_parquet_fp = os.path.join(cache_dir, f'{tag}.parquet')

        # Workaround for https://issues.apache.org/jira/browse/ARROW-1614
        # Currently, Arrow does not support storing multi-dimensional arrays / tensors.
        # When we write a column of tensors to disk, we need to first flatten it into a
        # 1D array, which we will then reshape back when we read the data at train time.
        features = get_combined_features(config)
        for feature in features:
            name = feature[NAME]
            proc_column = feature[PROC_COLUMN]
            reshape = training_set_metadata[name].get('reshape')
            if reshape is not None:
                dataset[proc_column] = self.map_objects(dataset[proc_column], lambda x: x.reshape(-1))

        os.makedirs(dataset_parquet_fp, exist_ok=True)
        dataset.to_parquet(dataset_parquet_fp,
                           engine='pyarrow',
                           write_index=False,
                           schema='infer')

        dataset_parquet_url = 'file://' + os.path.abspath(dataset_parquet_fp)
        training_set_metadata[DATASET_SPLIT_URL.format(tag)] = dataset_parquet_url

        return ParquetDataset(
            dataset_parquet_url,
            features,
            training_set_metadata
        )

    @property
    def array_lib(self):
        return da

    @property
    def df_lib(self):
        return dd

    @property
    def use_hdf5_cache(self):
        return False

    @property
    def parallelism(self):
        return self._parallelism
