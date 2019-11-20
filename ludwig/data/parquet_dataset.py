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

import pyarrow.parquet as pq


class ParquetDataset:
    def __init__(self, dataset, input_features, output_features, data_parquet_fp):
        self.dataset = dataset
        self.data_parquet_fp = data_parquet_fp
        self.parquet_file_reader = pq.ParquetFile(self.data_parquet_fp).reader

        self.size = self.parquet_file_reader.scan_contents()
        self.features = {}

        for feature in input_features + output_features:
            self.features[feature['name']] = feature

    def get(self, feature_name, idx=None):
        if idx is None:
            min_idx = 0
            max_idx = range(self.size)
        else:
            min_idx = min(idx)
            max_idx = max(idx)

        if self.data_parquet_fp is None:
            return self.dataset[feature_name][idx]

        if self.features[feature_name]['type'] in ('audio', 'image'):
            raise ValueError('Parquet dataset does not support audio/image '
                             'features yet')

        if self.features[feature_name]['type'] == 'text':
            feature_name = '{}_{}.list.item'.format(feature_name,
                                          self.features[feature_name]['level'])

        col_idx = self.parquet_file_reader.column_name_idx(feature_name)

        return self.parquet_file_reader.read_column(col_idx)[min_idx:max_idx].to_pylist()

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset
