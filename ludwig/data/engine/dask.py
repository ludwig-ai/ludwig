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

try:
    import dask
    import dask.array as da
    import dask.dataframe as dd
    import ray
    _LOADED = True
except:
    _LOADED = False

from ludwig.data.dataset.parquet import ParquetDataset
from ludwig.data.engine.base import DataProcessingEngine
from ludwig.utils.data_utils import DATASET_SPLIT_URL, replace_file_extension
from ludwig.utils.misc_utils import get_features


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

    def create_dataset(self, dataset, tag, config, training_set_metadata):
        tag = tag.lower()
        dataset_parquet_fp = replace_file_extension(dataset, f'.{tag}.parquet')

        os.makedirs(dataset_parquet_fp, exist_ok=True)
        dataset.to_parquet(dataset_parquet_fp,
                           engine='pyarrow',
                           write_index=False,
                           schema="infer")

        dataset_parquet_url = 'file://' + dataset_parquet_fp
        training_set_metadata[DATASET_SPLIT_URL.format(tag)] = dataset_parquet_url

        return ParquetDataset(
            dataset_parquet_url,
            get_features(config)
        )

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

    @property
    def use_hdf5_cache(self):
        return False
