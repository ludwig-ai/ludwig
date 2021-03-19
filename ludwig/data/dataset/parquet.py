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
import tensorflow as tf

from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset

from ludwig.constants import NAME, PROC_COLUMN
from ludwig.data.batcher.iterable import IterableBatcher
from ludwig.data.dataset.base import Dataset


class ParquetDataset(Dataset):
    def __init__(self, url, features, training_set_metadata):
        self.url = url
        self.training_set_metadata = training_set_metadata

        with make_batch_reader(self.url) as reader:
            self.size = reader.dataset.metadata.num_rows

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

    def initialize_batcher(self,
                           batch_size=128,
                           should_shuffle=True,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        cur_shard, shard_count = None, None
        if horovod:
            cur_shard, shard_count = horovod.rank(), horovod.size()

        reader = make_batch_reader(self.url,
                                   cur_shard=cur_shard,
                                   shard_count=shard_count,
                                   num_epochs=None)

        total_samples = reader.dataset.metadata.num_rows
        local_samples = int(total_samples / shard_count) if shard_count else total_samples

        dataset = make_petastorm_dataset(reader)
        dataset = dataset.unbatch()
        if should_shuffle:
            rows_per_piece = max([piece.get_metadata().num_rows for piece in reader.dataset.pieces])
            buffer_size = min(rows_per_piece, local_samples)
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)

        steps_per_epoch = int(local_samples / batch_size)

        batcher = IterableBatcher(self,
                                  dataset,
                                  steps_per_epoch,
                                  ignore_last=ignore_last)
        return batcher
