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
import contextlib
import math
import queue
import threading
from distutils.version import LooseVersion
from functools import lru_cache
from typing import Dict, Optional, Any, Iterator

import pandas as pd

import ray
from ray.data import from_dask
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.extensions import TensorDtype

from ludwig.data.batcher.base import Batcher
from ludwig.data.dataset.base import Dataset
from ludwig.features.base_feature import InputFeature, OutputFeature
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP
from ludwig.utils.misc_utils import get_proc_features
from ludwig.utils.types import DataFrame


_ray18 = LooseVersion(ray.__version__) >= LooseVersion("1.8")


class RayDataset(object):
    """ Wrapper around ray.data.Dataset. """

    def __init__(self, df: DataFrame, features: Dict[str, Dict], data_hdf5_fp: Optional[str]):
        self.ds = from_dask(df)
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp

        # TODO ray 1.8: convert to Tensors before shuffle
        # def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
        #     for c in features.keys():
        #         df[c] = df[c].astype(TensorDtype())
        #     return df
        # self.ds = self.ds.map_batches(to_tensors, batch_format="pandas")

    def pipeline(self, shuffle=True) -> DatasetPipeline:
        pipe = self.ds.repeat()
        if shuffle:
            if _ray18:
                pipe = pipe.random_shuffle_each_window()
            else:
                pipe = pipe.random_shuffle()
        return pipe

    def __len__(self):
        return self.ds.count()


class RayDatasetManager(object):
    def __init__(self, backend):
        self.backend = backend

    def create(self, dataset: DataFrame, config: Dict[str, Any], training_set_metadata: Dict[str, Any]):
        return RayDataset(
            dataset,
            get_proc_features(config),
            training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        )

    # TODO(travis): consider combining this with `create` when Petastorm is dropped
    def create_inference_dataset(
            self,
            dataset: DataFrame,
            tag: str,
            config: Dict[str, Any],
            training_set_metadata: Dict[str, Any]
    ):
        return self.create(dataset, config, training_set_metadata)

    def save(
            self,
            cache_path: str,
            dataset: DataFrame,
            config: Dict[str, Any],
            training_set_metadata: Dict[str, Any],
            tag: str
    ):
        # TODO(travis): optionally save dataset to Parquet for reuse
        return dataset

    def can_cache(self, skip_save_processed_input):
        # TODO(travis): enable caching
        # return self.backend.is_coordinator()
        return False

    @property
    def data_format(self):
        return 'parquet'


class RayDatasetShard(Dataset):
    def __init__(
            self,
            dataset_shard: DatasetPipeline,
            input_features: Dict[str, InputFeature],
            output_features: Dict[str, OutputFeature],
    ):
        self.dataset_shard = dataset_shard
        self.input_features = input_features
        self.output_features = output_features
        self.dataset_iter = dataset_shard.iter_datasets()

    @contextlib.contextmanager
    def initialize_batcher(self, batch_size=128,
                           should_shuffle=True,
                           shuffle_buffer_size=None,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        yield RayDatasetBatcher(
            self.dataset_iter,
            self.input_features,
            self.output_features,
            batch_size,
            self.size,
        )

    @lru_cache(1)
    def __len__(self):
        # TODO(travis): find way to avoid calling this, as it's expensive
        return next(self.dataset_iter).count()

    @property
    def size(self):
        return len(self)


class RayDatasetBatcher(Batcher):
    def __init__(
            self,
            dataset_epoch_iterator: Iterator[ray.data.Dataset],
            input_features: Dict[str, InputFeature],
            output_features: Dict[str, OutputFeature],
            batch_size: int,
            samples_per_epoch: int,
    ):
        self.dataset_epoch_iterator = dataset_epoch_iterator
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch

        self.columns = [
            f.proc_column for f in input_features.values()
        ] + [
            f.proc_column for f in output_features.values()
        ]

        self.dataset_batch_iter = None
        self._epoch = 0
        self._next_batch = None
        self._last_batch = False
        self._step = 0
        self._fetch_next_epoch()

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

    def _fetch_next_epoch(self):
        dataset = next(self.dataset_epoch_iterator)

        read_parallelism = 1
        if read_parallelism == 1:
            self.dataset_batch_iter = self._create_async_reader(dataset)
        elif read_parallelism > 1:
            self.dataset_batch_iter = self._create_async_parallel_reader(dataset, read_parallelism)
        else:
            # TODO: consider removing this. doesn't work currently and read performance seems generally
            #  very good with 1 parallelism
            self.dataset_batch_iter = self._create_sync_reader(dataset)

        self._step = 0
        self._fetch_next_batch()

    def _fetch_next_batch(self):
        if self.dataset_batch_iter is None:
            self._last_batch = True
            return

        self._last_batch = False
        try:
            self._next_batch = next(self.dataset_batch_iter)
        except StopIteration:
            self._last_batch = True

    def _create_sync_reader(self, dataset: ray.data.Dataset):
        columns = self.columns

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = df[c].astype(TensorDtype())
            return df

        def sync_read():
            for batch in dataset.map_batches(
                to_tensors, batch_format="pandas"
            ).iter_batches(
                prefetch_blocks=0,
                batch_size=self.batch_size,
                batch_format="pandas"
            ):
                yield {
                    c: batch[c].to_numpy() for c in self.columns
                }

        return sync_read()

    def _create_async_reader(self, dataset: ray.data.Dataset):
        q = queue.Queue(maxsize=100)

        columns = self.columns
        batch_size = self.batch_size

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = df[c].astype(TensorDtype())
            return df

        def producer():
            for batch in dataset.map_batches(
                    to_tensors,
                    batch_format="pandas"
            ).iter_batches(
                prefetch_blocks=0,
                batch_size=batch_size,
                batch_format="pandas"
            ):
                res = {
                    c: batch[c].to_numpy() for c in columns
                }
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

    def _create_async_parallel_reader(self, dataset: ray.data.Dataset, num_threads: int):
        q = queue.Queue(maxsize=100)

        columns = self.columns
        batch_size = self.batch_size

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = df[c].astype(TensorDtype())
            return df

        splits = dataset.split(n=num_threads)

        def producer(i):
            for batch in splits[i].map_batches(
                    to_tensors,
                    batch_format="pandas"
            ).iter_batches(
                prefetch_blocks=0,
                batch_size=batch_size,
                batch_format="pandas"
            ):
                res = {
                    c: batch[c].to_numpy() for c in columns
                }
                q.put(res)
            q.put(None)

        def async_parallel_read():
            threads = [threading.Thread(target=producer, args=(i,)) for i in range(num_threads)]
            for t in threads:
                t.start()

            active_threads = num_threads
            while True:
                batch = q.get(block=True)
                if batch is None:
                    active_threads -= 1
                    if active_threads == 0:
                        break
                yield batch

            for t in threads:
                t.join()

        return async_parallel_read()
