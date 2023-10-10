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
import logging
import math

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.data.batcher.base import Batcher

logger = logging.getLogger(__name__)


@DeveloperAPI
class RandomAccessBatcher(Batcher):
    def __init__(self, dataset, sampler, batch_size=128, ignore_last=False, augmentation_pipeline=None):
        # store our dataset as well
        self.dataset = dataset
        self.sampler = sampler
        self.sample_it = iter(self.sampler)

        self.ignore_last = ignore_last
        self.batch_size = batch_size
        self.total_size = len(sampler)
        self.augmentation_pipeline = augmentation_pipeline
        self.steps_per_epoch = self._compute_steps_per_epoch()
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

        sub_batch = {feature_name: self.dataset.get(feature_name, indices) for feature_name in self.dataset.features}

        if self.augmentation_pipeline:
            for feature_name, augmentations in self.augmentation_pipeline.items():
                logger.debug(f"RandomAccessBatcher applying augmentation pipeline to batch for feature {feature_name}")
                sub_batch[feature_name] = augmentations(torch.tensor(sub_batch[feature_name]))

        self.step += 1
        return sub_batch

    def last_batch(self):
        """Returns whether we've exhausted all batches for this epoch.

        If False, then there is at least 1 more batch available with next_batch().
        """
        # If our current index in the dataset exceeds the size of the dataset,
        # we've finished the epoch and can indicate that this is the last batch
        if self.index >= self.total_size:
            return True
        # This avoids the case where batch size > total size and no steps have been done.
        # For e.g., batch size = 128 but the dataset only has 100 rows.
        elif self.ignore_last and self.step:
            # index += batch_size after each epoch. So, if our current index in total dataset is 1 less than the total
            # dataset size, then the last batch will only have 1 row.
            # If this happens, we drop the last batch, unless batch_size is 1.
            if self.batch_size > 1 and self.index - self.total_size == -1:
                logger.info("Last batch in epoch only has 1 sample and will be dropped.")
                return True
        return False

    def set_epoch(self, epoch, batch_size):
        self.batch_size = batch_size
        self.steps_per_epoch = self._compute_steps_per_epoch()
        self.index = 0
        self.step = 0
        self.sampler.set_epoch(epoch)
        self.sample_it = iter(self.sampler)

    def _compute_steps_per_epoch(self):
        return int(math.ceil(self.total_size / self.batch_size))
