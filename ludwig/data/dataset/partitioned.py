#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
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
from ludwig.data.dataset.pandas import PandasDataset


class PartitionedDataset(object):
    def __init__(self, df, features, data_hdf5_fp):
        self.df = df
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp

    def get(self, col):
        return self.df[col]

    def map_dataset_partitions(self, fn, meta):
        def wrapped(partition):
            dataset = PandasDataset(partition, self.features, self.data_hdf5_fp)
            return fn(dataset)

        return self.df.map_partitions(wrapped, meta=meta)
