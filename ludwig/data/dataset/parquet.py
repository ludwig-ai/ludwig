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
import contextlib
import math
from os import read

import tensorflow as tf
from ludwig.data.dataset.pandas import PandasDataset
from ludwig.data.dataset.partitioned import PartitionedDataset
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset

from ludwig.constants import NAME, PROC_COLUMN
from ludwig.data.batcher.iterable import IterableBatcher
from ludwig.data.dataset.base import Dataset
from ludwig.utils.fs_utils import to_url
from ludwig.utils.misc_utils import get_combined_features, get_proc_features

from ray.data import read_parquet


class ParquetDataset(Dataset):
    def __init__(self, url, features, training_set_metadata):
        self.url = to_url(url)
        self.features = [feature[PROC_COLUMN] for feature in features]
        self.training_set_metadata = training_set_metadata

        with make_batch_reader(self.url) as reader:
            self.size = sum(piece.get_metadata().num_rows for piece in reader.dataset.pieces)

        self.reshape_features = {
            feature[PROC_COLUMN]: list((-1, *training_set_metadata[feature[NAME]]['reshape']))
            for feature in features
            if 'reshape' in training_set_metadata[feature[NAME]]
        }

    def get(self, feature_name, sample):
        t = getattr(sample, feature_name)
        reshape_dim = self.reshape_features.get(feature_name)
        if reshape_dim is not None:
            # When we read a 1D array from disk, we need to reshape it back to its
            # full dimensions.
            t = tf.reshape(t, reshape_dim)
        return t

    def __len__(self):
        return self.size

    @contextlib.contextmanager
    def initialize_batcher(self,
                           batch_size=128,
                           should_shuffle=True,
                           shuffle_buffer_size=None,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        cur_shard, shard_count = None, None
        if horovod:
            cur_shard, shard_count = horovod.rank(), horovod.size()

        with make_batch_reader(self.url,
                               cur_shard=cur_shard,
                               shard_count=shard_count,
                               num_epochs=None) as reader:
            total_samples = self.size
            local_samples = int(total_samples / shard_count) if shard_count else total_samples

            dataset = make_petastorm_dataset(reader)
            dataset = dataset.unbatch()
            if should_shuffle:
                rows_per_piece = max([piece.get_metadata().num_rows for piece in reader.dataset.pieces])
                buffer_size = shuffle_buffer_size or min(rows_per_piece, local_samples)
                dataset = dataset.shuffle(buffer_size)
            dataset = dataset.batch(batch_size)

            steps_per_epoch = math.ceil(local_samples / batch_size)

            batcher = IterableBatcher(self,
                                      dataset,
                                      steps_per_epoch,
                                      ignore_last=ignore_last)
            yield batcher


class ParquetDatasetManager(object):
    def __init__(self, backend):
        self.backend = backend

    def create(self, dataset, config, training_set_metadata):
        features = get_combined_features(config)
        return ParquetDataset(
            dataset,
            features,
            training_set_metadata
        )

    def create_inference_dataset(self, dataset, tag, config, training_set_metadata):
        if self.backend.df_engine.partitioned:
            print(f'GOES HERE!!')
            print(f'')
            # return read_parquet(dataset.url)
            return PartitionedDataset(
                dataset,
                get_proc_features(config),
                training_set_metadata.get(DATA_TRAIN_HDF5_FP)
            )
        else:
            return PandasDataset(
                dataset,
                get_proc_features(config),
                training_set_metadata.get(DATA_TRAIN_HDF5_FP)
            )

    def save(self, cache_path, dataset, config, training_set_metadata, tag):
        dataset_parquet_fp = cache_path

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
                dataset[proc_column] = self.backend.df_engine.map_objects(
                    dataset[proc_column],
                    lambda x: x.reshape(-1)
                )

        self.backend.df_engine.to_parquet(dataset, dataset_parquet_fp)
        return dataset_parquet_fp

    def can_cache(self, skip_save_processed_input):
        return self.backend.is_coordinator()

    @property
    def data_format(self):
        return 'parquet'
