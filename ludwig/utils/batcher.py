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
import math

import numpy as np

from ludwig.utils.data_utils import shuffle_dict_unison_inplace, shuffle_inplace


class Batcher(object):
    def __init__(self, dataset, batch_size=128, should_shuffle=True,
                 ignore_last=False):
        self.should_shuffle = should_shuffle

        # store our dataset as well
        self.dataset = dataset
        if should_shuffle:
            shuffle_inplace(self.dataset.get_dataset())

        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = dataset.size
        self.steps_per_epoch = int(math.ceil(self.total_size / self.batch_size))
        self.index = 0
        self.step = 0
        self.epoch = 0

    def next_batch(self):
        if self.last_batch():
            if self.should_shuffle:
                self.dataset = shuffle_dict_unison_inplace(self.dataset)
            self.reset()
            self.epoch += 1

        sub_batch = {}
        for features_name in self.dataset.features:
            sub_batch[features_name] = self.dataset.get(
                features_name,
                range(
                    self.index,
                    min(self.index + self.batch_size, self.total_size)
                )
            )

        self.index += self.batch_size
        self.step += 1
        return sub_batch

    def last_batch(self):
        return self.index >= self.total_size or (
                self.ignore_last and
                self.index + self.batch_size >= self.total_size)

    def reset(self):
        self.index = 0
        self.step = 0


class BucketedBatcher(object):
    def __init__(self, dataset, bucketing_field, batch_size=128, buckets=10,
                 should_shuffle=True, ignore_last=False,
                 should_trim=False, trim_side='right'):
        self.should_shuffle = should_shuffle
        self.bucketing_field = bucketing_field
        self.should_trim = should_trim
        self.trim_side = trim_side

        # store our dataset as well
        self.dataset = dataset

        field = dataset.get_dataset()[bucketing_field]
        field_lengths = np.apply_along_axis(lambda x: np.sign(x).sum(), 1,
                                            field)
        sorted_idcs = np.argsort(field_lengths)
        self.buckets_idcs = []
        datapoints_per_bucket = len(field) // buckets
        for b in range(buckets):
            start = datapoints_per_bucket * b
            end = datapoints_per_bucket * (b + 1) if b < buckets - 1 else len(
                sorted_idcs)
            self.buckets_idcs.append(sorted_idcs[start:end])

        if should_shuffle:
            self.shuffle(self.buckets_idcs)

        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = min(map(len, dataset.get_dataset().values()))
        self.bucket_sizes = np.array([x for x in map(len, self.buckets_idcs)])
        self.steps_per_epoch = int(
            np.asscalar(np.sum(np.ceil(self.bucket_sizes / self.batch_size))))
        self.indices = np.array([0] * buckets)
        self.step = 0
        self.epoch = 0

    def shuffle(self, buckets_idcs):
        for i in range(len(buckets_idcs)):
            np.random.shuffle(buckets_idcs[i])

    def next_batch(self):
        if self.last_batch():
            if self.should_shuffle:
                self.shuffle(self.buckets_idcs)
            self.reset()
            self.epoch += 1

        if self.ignore_last:
            idcs_below_size = self.indices + self.batch_size < self.bucket_sizes
        else:
            idcs_below_size = self.indices < self.bucket_sizes
        i = np.random.choice(
            np.arange(0, len(self.buckets_idcs))[idcs_below_size])

        selected_bucket = self.buckets_idcs[i]
        selected_idcs = selected_bucket[
                        self.indices[i]:self.indices[i] + self.batch_size]

        sub_batch = {}
        for key in self.dataset.get_dataset():
            if key == self.bucketing_field and self.should_trim:
                selected_samples = self.dataset.get(key, selected_idcs)
                max_length = np.sign(selected_samples).sum(axis=1).max()
                if self.trim_side == 'right':
                    sub_batch[key] = selected_samples[:, :max_length]
                elif self.trim_side == 'left':
                    sub_batch[key] = selected_samples[:, -max_length:]
                else:
                    raise ValueError('Invalid trim side:', self.trim_side)

            else:
                sub_batch[key] = self.dataset.get(key, selected_idcs)

        self.indices[i] += self.batch_size
        self.step += 1
        return sub_batch

    def last_batch(self):
        return not np.any(self.indices < self.bucket_sizes) \
               or (self.ignore_last and
                   not np.any(
                       self.indices + self.batch_size < self.bucket_sizes
                   ))

    def reset(self):
        self.indices = np.array([0] * len(self.buckets_idcs))
        self.step = 0


class DistributedBatcher(object):
    def __init__(self, dataset, partition_number, horovod, batch_size=128,
                 should_shuffle=True, ignore_last=False):
        self.should_shuffle = should_shuffle

        # store our dataset as well
        partition_size = dataset.size // horovod.size()
        if partition_number == horovod.size() - 1:
            self.partition = (partition_size * partition_number, dataset.size)
        else:
            self.partition = (partition_size * partition_number,
                              partition_size * (partition_number + 1))
        self.dataset = dataset
        if should_shuffle:
            shuffle_inplace(self.dataset.get_dataset())

        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = self.partition[1] - self.partition[0]
        self.steps_per_epoch = int(math.ceil(self.total_size / self.batch_size))
        self.index = self.partition[0]
        self.max_index = self.partition[1]
        self.step = 0
        self.epoch = 0

    def next_batch(self):
        if self.last_batch():
            if self.should_shuffle:
                self.dataset = shuffle_dict_unison_inplace(
                    self.dataset,
                    np.random.RandomState(self.epoch)
                )
            self.reset()
            self.epoch += 1

        sub_batch = {}
        for features_name in self.dataset.features:
            sub_batch[features_name] = self.dataset.get(
                features_name,
                range(
                    self.index,
                    min(self.index + self.batch_size, self.max_index)
                )
            )

        self.index += self.batch_size
        self.step += 1
        return sub_batch

    def last_batch(self):
        return self.index >= self.max_index or (
                self.ignore_last and
                self.index + self.batch_size >= self.max_index)

    def reset(self):
        self.index = self.partition[0]
        self.step = 0
