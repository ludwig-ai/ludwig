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
import math
import queue
import threading
from functools import lru_cache
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from pyarrow.fs import FSSpecHandler, PyFileSystem
from ray.data import Dataset as RayNativeDataset
from ray.data import read_parquet
from ray.data.extensions import TensorArray

from ludwig.backend.base import Backend
from ludwig.constants import BINARY, CATEGORY, NAME, NUMBER, TYPE
from ludwig.data.batcher.base import Batcher
from ludwig.data.dataset.base import Dataset, DatasetManager
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP, DATA_TRAIN_PARQUET_FP
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import get_fs_and_path
from ludwig.utils.misc_utils import get_proc_features
from ludwig.utils.types import DataFrame, Series

_SCALAR_TYPES = {BINARY, CATEGORY, NUMBER}


def cast_as_tensor_dtype(series: Series) -> Series:
    return TensorArray(series)


def read_remote_parquet(path: str):
    fs, path = get_fs_and_path(path)
    return read_parquet(path, filesystem=PyFileSystem(FSSpecHandler(fs)))


class RayDataset(Dataset):
    """Wrapper around ray.data.Dataset."""

    def __init__(
        self,
        df: str | DataFrame,
        features: dict[str, dict],
        training_set_metadata: dict[str, Any],
        backend: Backend,
    ):
        self.df_engine = backend.df_engine
        self.ds = self.df_engine.to_ray_dataset(df) if not isinstance(df, str) else read_remote_parquet(df)
        self.features = features
        self.training_set_metadata = training_set_metadata
        self.data_hdf5_fp = training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        self.data_parquet_fp = training_set_metadata.get(DATA_TRAIN_PARQUET_FP)

    def to_ray_dataset(
        self,
        shuffle: bool = True,
        shuffle_seed: int = default_random_seed,
    ) -> RayNativeDataset:
        """Returns a ray.data.Dataset, optionally shuffled.

        In modern Ray (2.5+), datasets use lazy execution by default, so there's no need for explicit windowing or
        pipelining.
        """
        ds = self.ds
        if shuffle:
            ds = ds.random_shuffle(seed=shuffle_seed)
        return ds

    @contextlib.contextmanager
    def initialize_batcher(self, batch_size=128, should_shuffle=True, seed=0, ignore_last=False):
        ds = self.ds
        if should_shuffle:
            ds = ds.random_shuffle(seed=seed)
        yield RayDatasetBatcher(
            ds,
            self.features,
            self.training_set_metadata,
            batch_size,
            self.size,
        )

    def __len__(self):
        return self.ds.count()

    @property
    def size(self):
        return len(self)

    @property
    def in_memory_size_bytes(self):
        return self.ds.size_bytes() if self.ds is not None else 0

    def to_df(self):
        return self.df_engine.from_ray_dataset(self.ds)


class RayDatasetManager(DatasetManager):
    def __init__(self, backend):
        self.backend = backend

    def create(self, dataset: str | DataFrame, config: dict[str, Any], training_set_metadata: dict[str, Any]):
        return RayDataset(dataset, get_proc_features(config), training_set_metadata, self.backend)

    def save(
        self,
        cache_path: str,
        dataset: DataFrame,
        config: dict[str, Any],
        training_set_metadata: dict[str, Any],
        tag: str,
    ):
        self.backend.df_engine.to_parquet(dataset, cache_path)
        return cache_path

    def can_cache(self, skip_save_processed_input):
        return not skip_save_processed_input

    @property
    def data_format(self):
        return "parquet"


class RayDatasetShard(Dataset):
    """Wraps a Ray DataIterator (from ray.train.get_dataset_shard) for distributed training."""

    def __init__(
        self,
        dataset_shard,
        features: dict[str, dict],
        training_set_metadata: dict[str, Any],
    ):
        self.dataset_shard = dataset_shard
        self.features = features
        self.training_set_metadata = training_set_metadata

    @contextlib.contextmanager
    def initialize_batcher(self, batch_size=128, should_shuffle=True, seed=0, ignore_last=False):
        yield RayDatasetShardBatcher(
            self.dataset_shard,
            self.features,
            self.training_set_metadata,
            batch_size,
            self.size,
        )

    @lru_cache(1)
    def __len__(self):
        # TODO(travis): find way to avoid calling this, as it's expensive
        # DataIterator doesn't have a direct count method; use iter to count
        count = 0
        for batch in self.dataset_shard.iter_batches(batch_size=4096, batch_format="pandas"):
            count += len(batch)
        return count

    @property
    def size(self):
        return len(self)


class _BaseBatcher(Batcher):
    """Shared batching logic for preparing batches from pandas DataFrames."""

    def __init__(
        self,
        features: dict[str, dict],
        training_set_metadata: dict[str, Any],
        batch_size: int,
        samples_per_epoch: int,
    ):
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.training_set_metadata = training_set_metadata

        self.features = features
        self.columns = list(features.keys())
        self.reshape_map = {
            proc_column: training_set_metadata[feature[NAME]].get("reshape")
            for proc_column, feature in features.items()
        }

        self.dataset_batch_iter = None
        self._epoch = 0
        self._next_batch = None
        self._last_batch = False
        self._step = 0

    def next_batch(self):
        if self.last_batch():
            raise StopIteration()

        batch = self._next_batch
        self._fetch_next_batch()
        self._step += 1
        return batch

    def last_batch(self):
        return self._last_batch

    def set_epoch(self, epoch, batch_size):
        self.batch_size = batch_size
        if epoch != self._epoch:
            self._fetch_next_epoch()
            self._epoch = epoch

    @property
    def step(self):
        return self._step

    @property
    def steps_per_epoch(self):
        return math.ceil(self.samples_per_epoch / self.batch_size)

    def _fetch_next_batch(self):
        if self.dataset_batch_iter is None:
            self._last_batch = True
            return

        self._last_batch = False
        try:
            self._next_batch = next(self.dataset_batch_iter)
        except StopIteration:
            self._last_batch = True

    def _fetch_next_epoch(self):
        raise NotImplementedError

    def _to_tensors_fn(self):
        columns = self.columns
        features = self.features

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                # do not convert scalar columns: https://github.com/ray-project/ray/issues/20825
                if features[c][TYPE] not in _SCALAR_TYPES:
                    df[c] = cast_as_tensor_dtype(df[c])
                elif features[c][TYPE] == BINARY:
                    df[c] = df[c].astype(np.bool_)
            return df

        return to_tensors

    def _prepare_batch(self, batch: pd.DataFrame) -> dict[str, np.ndarray]:
        res = {}
        for c in self.columns:
            if self.features[c][TYPE] not in _SCALAR_TYPES:
                res[c] = np.stack(batch[c].values)
            else:
                res[c] = batch[c].to_numpy()

        for c in self.columns:
            reshape = self.reshape_map.get(c)
            if reshape is not None:
                res[c] = res[c].reshape((-1, *reshape))
        return res


class RayDatasetBatcher(_BaseBatcher):
    """Batcher for a full ray.data.Dataset (used by non-distributed/local Ray training)."""

    def __init__(
        self,
        dataset: RayNativeDataset,
        features: dict[str, dict],
        training_set_metadata: dict[str, Any],
        batch_size: int,
        samples_per_epoch: int,
    ):
        self.dataset = dataset
        super().__init__(features, training_set_metadata, batch_size, samples_per_epoch)
        self._fetch_next_epoch()

    def _fetch_next_epoch(self):
        """Create an async reader over the dataset for one epoch."""
        self.dataset_batch_iter = self._create_async_reader(self.dataset)
        self._step = 0
        self._fetch_next_batch()

    def _create_async_reader(self, dataset: RayNativeDataset):
        q = queue.Queue(maxsize=100)
        batch_size = self.batch_size
        to_tensors = self._to_tensors_fn()

        def producer():
            for batch in dataset.map_batches(to_tensors, batch_format="pandas").iter_batches(
                prefetch_batches=1, batch_size=batch_size, batch_format="pandas"
            ):
                res = self._prepare_batch(batch)
                q.put(res)
            q.put(None)

        def async_read():
            t = threading.Thread(target=producer)
            t.start()
            while True:
                batch = q.get(block=True)
                if batch is None:
                    break
                yield batch
            t.join()

        return async_read()


class RayDatasetShardBatcher(_BaseBatcher):
    """Batcher for a Ray DataIterator shard (used in distributed training workers)."""

    def __init__(
        self,
        data_iterator,
        features: dict[str, dict],
        training_set_metadata: dict[str, Any],
        batch_size: int,
        samples_per_epoch: int,
    ):
        self.data_iterator = data_iterator
        super().__init__(features, training_set_metadata, batch_size, samples_per_epoch)
        self._fetch_next_epoch()

    def _fetch_next_epoch(self):
        """Create an async reader from the DataIterator for one epoch."""
        self.dataset_batch_iter = self._create_async_reader()
        self._step = 0
        self._fetch_next_batch()

    def _create_async_reader(self):
        q = queue.Queue(maxsize=100)
        batch_size = self.batch_size
        to_tensors = self._to_tensors_fn()

        def producer():
            for batch in self.data_iterator.iter_batches(
                batch_size=batch_size,
                batch_format="pandas",
                prefetch_batches=1,
            ):
                batch = to_tensors(batch)
                res = self._prepare_batch(batch)
                q.put(res)
            q.put(None)

        def async_read():
            t = threading.Thread(target=producer)
            t.start()
            while True:
                batch = q.get(block=True)
                if batch is None:
                    break
                yield batch
            t.join()

        return async_read()
