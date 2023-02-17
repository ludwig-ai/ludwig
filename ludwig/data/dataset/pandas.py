#! /usr/bin/env python
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
import contextlib
from typing import Iterable, Optional

import numpy as np
from pandas import DataFrame

from ludwig.constants import PREPROCESSING, TRAINING
from ludwig.data.batcher.random_access import RandomAccessBatcher
from ludwig.data.dataset.base import Dataset, DatasetManager
from ludwig.data.sampler import DistributedSampler
from ludwig.distributed import DistributedStrategy
from ludwig.features.base_feature import BaseFeature
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP, save_hdf5
from ludwig.utils.dataframe_utils import from_numpy_dataset, to_numpy_dataset
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import download_h5
from ludwig.utils.misc_utils import get_proc_features


class PandasDataset(Dataset):
    def __init__(self, dataset, features, data_hdf5_fp):
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp
        self.size = len(dataset)
        self.dataset = to_numpy_dataset(dataset)

    def to_df(self, features: Optional[Iterable[BaseFeature]] = None) -> DataFrame:
        """Convert the dataset to a Pandas DataFrame."""
        if features:
            return from_numpy_dataset({feature.feature_name: self.dataset[feature.proc_column] for feature in features})
        return from_numpy_dataset(self.dataset)

    def get(self, proc_column, idx=None):
        if idx is None:
            idx = range(self.size)
        if (
            self.data_hdf5_fp is None
            or PREPROCESSING not in self.features[proc_column]
            or "in_memory" not in self.features[proc_column]["preprocessing"]
        ):
            return self.dataset[proc_column][idx]
        if self.features[proc_column][PREPROCESSING]["in_memory"]:
            return self.dataset[proc_column][idx]

        sub_batch = self.dataset[proc_column][idx]

        indices = np.empty((3, len(sub_batch)), dtype=np.int64)
        indices[0, :] = sub_batch
        indices[1, :] = np.arange(len(sub_batch))
        indices = indices[:, np.argsort(indices[0])]

        with download_h5(self.data_hdf5_fp) as h5_file:
            im_data = h5_file[proc_column + "_data"][indices[0, :], :, :]
        indices[2, :] = np.arange(len(sub_batch))
        indices = indices[:, np.argsort(indices[1])]
        return im_data[indices[2, :]]

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return self.size

    @property
    def in_memory_size_bytes(self):
        df = self.to_df()
        return df.memory_usage(deep=True).sum() if df is not None else 0

    @contextlib.contextmanager
    def initialize_batcher(
        self,
        batch_size: int = 128,
        should_shuffle: bool = True,
        random_seed: int = default_random_seed,
        ignore_last: bool = False,
        distributed: DistributedStrategy = None,
        augmentation_pipeline=None,
    ):
        sampler = DistributedSampler(
            len(self), shuffle=should_shuffle, random_seed=random_seed, distributed=distributed
        )
        batcher = RandomAccessBatcher(
            self,
            sampler,
            batch_size=batch_size,
            ignore_last=ignore_last,
            augmentation_pipeline=augmentation_pipeline,
        )
        yield batcher


class PandasDatasetManager(DatasetManager):
    def __init__(self, backend):
        self.backend = backend

    def create(self, dataset, config, training_set_metadata):
        return PandasDataset(dataset, get_proc_features(config), training_set_metadata.get(DATA_TRAIN_HDF5_FP))

    def save(self, cache_path, dataset, config, training_set_metadata, tag):
        save_hdf5(cache_path, dataset)
        if tag == TRAINING:
            training_set_metadata[DATA_TRAIN_HDF5_FP] = cache_path
        return dataset

    def can_cache(self, skip_save_processed_input):
        return self.backend.is_coordinator() and not skip_save_processed_input

    @property
    def data_format(self):
        return "hdf5"
