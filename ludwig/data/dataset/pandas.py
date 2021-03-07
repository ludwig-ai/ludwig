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

from ludwig.constants import PREPROCESSING
from ludwig.data.batcher.random_access import RandomAccessBatcher
from ludwig.data.dataset.base import Dataset
from ludwig.data.sampler import DistributedSampler
from ludwig.utils.data_utils import to_numpy_dataset


class PandasDataset(Dataset):
    def __init__(self, dataset, features, data_hdf5_fp):
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp
        self.size = len(dataset)
        self.dataset = to_numpy_dataset(dataset)

    def get(self, proc_column, idx=None):
        if idx is None:
            idx = range(self.size)
        if (self.data_hdf5_fp is None or
                PREPROCESSING not in self.features[proc_column] or
                'in_memory' not in self.features[proc_column][
                    'preprocessing']):
            return self.dataset[proc_column][idx]
        if self.features[proc_column][PREPROCESSING]['in_memory']:
            return self.dataset[proc_column][idx]

        sub_batch = self.dataset[proc_column][idx]

        indices = np.empty((3, len(sub_batch)), dtype=np.int64)
        indices[0, :] = sub_batch
        indices[1, :] = np.arange(len(sub_batch))
        indices = indices[:, np.argsort(indices[0])]

        with h5py.File(self.data_hdf5_fp, 'r') as h5_file:
            im_data = h5_file[proc_column + '_data'][indices[0, :], :, :]
        indices[2, :] = np.arange(len(sub_batch))
        indices = indices[:, np.argsort(indices[1])]
        return im_data[indices[2, :]]

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return self.size

    def initialize_batcher(self, batch_size=128,
                           should_shuffle=True,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        sampler = DistributedSampler(len(self),
                                     shuffle=should_shuffle,
                                     seed=seed,
                                     horovod=horovod)
        batcher = RandomAccessBatcher(self,
                                      sampler,
                                      batch_size=batch_size,
                                      ignore_last=ignore_last)
        return batcher
