#! /usr/bin/env python
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

import math

import numpy as np

from ludwig.distributed import DistributedStrategy
from ludwig.utils.defaults import default_random_seed


class DistributedSampler:
    """Adapted from `torch.utils.data.distributed.DistributedSampler`."""

    def __init__(
        self,
        dataset_size: int,
        shuffle: bool = True,
        random_seed: int = default_random_seed,
        distributed: DistributedStrategy = None,
    ):
        self.dataset_size = dataset_size
        self.num_replicas = distributed.size() if distributed else 1
        self.rank = distributed.rank() if distributed else 0
        self.epoch = 0
        self.num_samples = int(math.ceil(self.dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.random_seed = random_seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = np.random.RandomState(seed=self.random_seed + self.epoch).permutation(self.dataset_size).tolist()
        else:
            indices = list(range(self.dataset_size))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Sets the epoch for this sampler.

        When `shuffle=True`, this ensures all replicas use a different random ordering
        for each epoch. Otherwise, the next iteration of this sampler will yield the same ordering.

        :param epoch: (int) epoch number
        """
        self.epoch = epoch
