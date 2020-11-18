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
from ludwig.data.batcher.base import Batcher


class IterableBatchProvider(Batcher):
    def __init__(self,
                 dataset,
                 iterable_dataset,
                 batch_size,
                 steps_per_epoch,
                 shuffle_buffer_size,
                 ignore_last=False):

        if shuffle_buffer_size > 0:
            iterable_dataset = iterable_dataset.shuffle(shuffle_buffer_size)
        iterable_dataset = iterable_dataset.batch(batch_size)

        self.dataset = dataset
        self.data_it = iter(iterable_dataset)

        self.ignore_last = ignore_last
        self.steps_per_epoch = steps_per_epoch
        self.step = 0

    def next_batch(self):
        if self.last_batch():
            raise StopIteration()

        sub_batch = {}
        batch = next(self.data_it)
        for features_name in self.dataset.features:
            sub_batch[features_name] = self.dataset.get(
                features_name,
                batch
            )

        self.step += 1
        return sub_batch

    def last_batch(self):
        return self.step >= self.steps_per_epoch or (
                self.ignore_last and
                self.step + 1 >= self.steps_per_epoch)

    def set_epoch(self, epoch):
        self.step = 0
