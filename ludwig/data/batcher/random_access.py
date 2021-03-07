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

from ludwig.data.batcher.base import Batcher


class RandomAccessBatcher(Batcher):
    def __init__(self, dataset, sampler,
                 batch_size=128,
                 ignore_last=False):
        # store our dataset as well
        self.dataset = dataset
        self.sampler = sampler
        self.sample_it = iter(self.sampler)

        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = len(sampler)
        self.steps_per_epoch = int(
            math.ceil(self.total_size / self.batch_size))
        self.index = 0
        self.step = 0

    def next_batch(self):
        if self.last_batch():
            raise StopIteration()

        indices = []
        for _ in range(self.batch_size):
            try:
                indices.append(next(self.sample_it))
                self.index += 1
            except StopIteration:
                break

        sub_batch = {}
        for features_name in self.dataset.features:
            sub_batch[features_name] = self.dataset.get(
                features_name,
                indices
            )

        self.step += 1
        return sub_batch

    def last_batch(self):
        return self.index >= self.total_size or (
                self.ignore_last and
                self.index + self.batch_size >= self.total_size)

    def set_epoch(self, epoch):
        self.index = 0
        self.step = 0
        self.sampler.set_epoch(epoch)
        self.sample_it = iter(self.sampler)
