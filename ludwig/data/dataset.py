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
import h5py
import numpy as np


class Dataset:
    def __init__(self, dataset, input_features, output_features, data_hdf5_fp):
        self.dataset = dataset

        self.size = min(map(len, self.dataset.values()))

        self.input_features = {}
        for feature in input_features:
            feature_name = feature['name']
            self.input_features[feature_name] = feature
        self.output_features = {}
        for feature in output_features:
            feature_name = feature['name']
            self.output_features[feature_name] = feature
        self.features = self.input_features.copy()
        self.features.update(self.output_features)
        self.data_hdf5_fp = data_hdf5_fp

    def get(self, feature_name, idx=None):
        if idx is None:
            idx = range(self.size)
        if (self.data_hdf5_fp is None or
                'preprocessing' not in self.features[feature_name] or
                'in_memory' not in self.features[feature_name]['preprocessing']):
            return self.dataset[feature_name][idx]
        if self.features[feature_name]['preprocessing']['in_memory']:
            return self.dataset[feature_name][idx]

        sub_batch = self.dataset[feature_name][idx]

        indices = np.empty((3, len(sub_batch)), dtype=np.int64)
        indices[0, :] = sub_batch
        indices[1, :] = np.arange(len(sub_batch))
        indices = indices[:, np.argsort(indices[0])]

        with h5py.File(self.data_hdf5_fp, 'r') as h5_file:
            im_data = h5_file[feature_name + '_data'][indices[0, :], :, :]
        indices[2, :] = np.arange(len(sub_batch))
        indices = indices[:, np.argsort(indices[1])]
        return im_data[indices[2, :]]

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset
