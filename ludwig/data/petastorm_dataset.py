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
from ludwig.utils.data_utils import text_feature_data_field
from petastorm import make_reader
from petastorm.tf_utils import make_petastorm_dataset


class PetaStormDataset:
    def __init__(self, input_features, output_features, data_parquet_fp):
        self.reader = make_reader(data_parquet_fp)
        # self.dataset = make_petastorm_dataset(self.reader)
        self.size = self.get_size()
        self.data_parquet_fp = data_parquet_fp

        self.input_features = {}
        for feature in input_features:

            if feature['type'] == 'text':
                feature_name = text_feature_data_field(feature)
            else:
                feature_name = feature['name']

            self.input_features[feature_name] = feature

        self.output_features = {}
        for feature in output_features:
            if feature['type'] == 'text':
                feature_name = text_feature_data_field(feature)
            else:
                feature_name = feature['name']

            self.output_features[feature_name] = feature

        self.features = self.input_features.copy()
        self.features.update(self.output_features)

    def next(self):
        return self.reader.next()

    def next_batch(self, batch_size):
        # return self.dataset.batch(batch_size, drop_remainder=drop_last)
        return [self.reader.next() for _ in range(batch_size)]

    def get_size(self):
        num_rows = 0
        while True:
            try:
                _ = self.reader.next()
                num_rows += 1
            except StopIteration:
                return num_rows

        return num_rows

    def shuffle(self, buffer_size):
        self.dataset.shuffle(buffer_size)

    def get_dataset(self):
        return self.dataset

    def reset(self):
        self.reader.reset()

    def set_dataset(self, dataset):
        self.dataset = dataset
