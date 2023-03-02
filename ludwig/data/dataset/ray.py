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
import logging
import math
import queue
import threading
from functools import lru_cache
from typing import Dict, Iterator, Literal, Optional, Union

import numpy as np
import pandas as pd
import ray
import torch
from packaging import version
from pyarrow.fs import FSSpecHandler, PyFileSystem
from ray.data import read_parquet
from ray.data.dataset_pipeline import DatasetPipeline

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend.base import Backend
from ludwig.constants import BINARY, CATEGORY, NAME, NUMBER, TYPE
from ludwig.data.batcher.base import Batcher
from ludwig.data.dataset.base import Dataset, DatasetManager
from ludwig.distributed import DistributedStrategy
from ludwig.types import FeatureConfigDict, ModelConfigDict, TrainingSetMetadataDict
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP, DATA_TRAIN_PARQUET_FP, from_numpy_dataset, to_numpy_dataset
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.error_handling_utils import default_retry
from ludwig.utils.fs_utils import get_fs_and_path
from ludwig.utils.misc_utils import get_proc_features
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)

_ray_230 = version.parse(ray.__version__) >= version.parse("2.3.0")

_SCALAR_TYPES = {BINARY, CATEGORY, NUMBER}


@DeveloperAPI
@default_retry()
def read_remote_parquet(path: str):
    fs, path = get_fs_and_path(path)
    return read_parquet(path, filesystem=PyFileSystem(FSSpecHandler(fs)))


@DeveloperAPI
class RayDataset(Dataset):
    """Wrapper around ray.data.Dataset.

    Args:
        df: The data to wrap
        features: Feature-level config indexed by feature name
        training_set_metadata: Additional training set information
        backend: The local/distributed compute coordinator
        window_size_bytes: The requested size of a dataset window in bytes. If "auto", sets the window size relative to
            the dataset size and object store size. If not specified, no windowing will occur.
    """

    def __init__(
        self,
        df: Union[str, DataFrame],
        features: Dict[str, FeatureConfigDict],
        training_set_metadata: TrainingSetMetadataDict,
        backend: Backend,
        window_size_bytes: Optional[Union[int, Literal["auto"]]] = None,
    ):
        self.df_engine = backend.df_engine
        self.ds = self.df_engine.to_ray_dataset(df) if not isinstance(df, str) else read_remote_parquet(df)
        self.features = features
        self.training_set_metadata = training_set_metadata
        self.data_hdf5_fp = training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        self.data_parquet_fp = training_set_metadata.get(DATA_TRAIN_PARQUET_FP)
        self._processed_data_fp = df if isinstance(df, str) else None
        self.window_size_bytes = self.get_window_size_bytes(window_size_bytes)

    def get_window_size_bytes(self, window_size_bytes: Optional[Union[int, Literal["auto"]]] = None) -> int:
        """Return this dataset's window size in bytes, or translate auto-windowing into bytes."""
        # If user has specified a window size, use it as-is.
        if isinstance(window_size_bytes, int):
            return window_size_bytes

        # If the user requests auto window sizing and the dataset is large,
        # set the window size to `<available memory> // 5`.
        elif window_size_bytes == "auto":
            ds_memory_size = self.in_memory_size_bytes
            cluster_memory_size = ray.cluster_resources()["object_store_memory"]
            if ds_memory_size > cluster_memory_size // 5:
                # TODO: Add link to windowing docs.
                logger.info(
                    "In-memory dataset size is greater than 20%% of object store memory. "
                    "Enabling windowed shuffling of data to prevent chances of OOMs. "
                )
                if _ray_230:
                    # In Ray nightly (>= 2.3), window size is specified as either -1 or a percentage
                    # from 0 to 1. Default to always using 20% of object store memory.
                    return 0.2
                return int(cluster_memory_size // 5)

        # By default, set to -1 so that an infinite window size
        # will be used which effectively results in bulk data ingestion
        return -1

    @contextlib.contextmanager
    def initialize_batcher(
        self,
        batch_size=128,
        should_shuffle=True,
        random_seed=0,
        ignore_last=False,
        distributed=None,
        augmentation_pipeline=None,
    ):
        yield RayDatasetBatcher(
            self.ds.repeat().iter_datasets(),
            self.features,
            self.training_set_metadata,
            batch_size,
            self.size,
            ignore_last,
            augmentation_pipeline=augmentation_pipeline,
        )

    def __len__(self):
        return self.ds.count()

    @property
    def size(self):
        return len(self)

    @property
    def processed_data_fp(self) -> Optional[str]:
        return self._processed_data_fp

    @property
    def in_memory_size_bytes(self):
        """Memory size may be unknown, so return 0 incase size_bytes() returns None
        https://docs.ray.io/en/releases-1.12.1/_modules/ray/data/dataset.html#Dataset.size_bytes."""
        return self.ds.size_bytes() if self.ds is not None else 0

    def to_df(self):
        return self.df_engine.from_ray_dataset(self.ds)

    def repartition(self, num_blocks: int):
        """Repartition the dataset into the specified number of blocks.

        This operation occurs in place and overwrites `self.ds` with a
        new repartitioned dataset.
        Args:
            num_blocks: Number of blocks in the repartitioned data.
        """
        self.ds = self.ds.repartition(num_blocks=num_blocks)


@DeveloperAPI
class RayDatasetManager(DatasetManager):
    def __init__(self, backend):
        self.backend = backend

    def create(
        self,
        dataset: Union[str, DataFrame],
        config: ModelConfigDict,
        training_set_metadata: TrainingSetMetadataDict,
    ) -> "RayDataset":
        """Create a new Ray dataset with config."""
        window_size_bytes = self.backend._data_loader_kwargs.get("window_size_bytes", None)
        return RayDataset(
            dataset, get_proc_features(config), training_set_metadata, self.backend, window_size_bytes=window_size_bytes
        )

    def save(
        self,
        cache_path: str,
        dataset: DataFrame,
        config: ModelConfigDict,
        training_set_metadata: TrainingSetMetadataDict,
        tag: str,
    ):
        self.backend.df_engine.to_parquet(dataset, cache_path)
        return cache_path

    def can_cache(self, skip_save_processed_input):
        return not skip_save_processed_input

    @property
    def data_format(self):
        return "parquet"


@DeveloperAPI
class RayDatasetShard(Dataset):
    def __init__(
        self,
        dataset_shard: DatasetPipeline,
        features: Dict[str, FeatureConfigDict],
        training_set_metadata: TrainingSetMetadataDict,
    ):
        self.dataset_shard = dataset_shard
        self.features = features
        self.training_set_metadata = training_set_metadata
        self.create_epoch_iter()

    def create_epoch_iter(self) -> None:
        if _ray_230:
            # In Ray >= 2.3, session.get_dataset_shard() returns a DatasetIterator object.
            if isinstance(self.dataset_shard, ray.data.DatasetIterator):
                if hasattr(self.dataset_shard, "_base_dataset_pipeline"):
                    # Dataset shard is a DatasetIterator that was created from a DatasetPipeline object.
                    # Retrieve the base object that was used to create the DatasetIterator so that we can
                    # create the iter_epochs() like in Ray <= 2.2.
                    self.epoch_iter = self.dataset_shard._base_dataset_pipeline.iter_epochs()
                    return
        else:
            # In Ray <= 2.2, session.get_dataset_shard() returns a DatasetPipeline object.
            if isinstance(self.dataset_shard, DatasetPipeline):
                # Dataset shard is a DatasetPipeline during training. The Ray Dataset is converted to a
                # DatasetPipeline by the DatasetConfig in the Trainer and is available in the train_fn
                self.epoch_iter = self.dataset_shard.iter_epochs()
                return

        # Here, dataset shard is a RayDataset object during auto batch size tuning or learning rate tuning
        # since it does not come from within the RayTrainer's train_fn.
        # Convert Ray Dataset to a DatasetPipeline object before enabling epoch iteration
        # In this scenario, there is no need to worry about windowing, shuffling etc.
        self.epoch_iter = self.dataset_shard.repeat().iter_epochs()

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
        yield RayDatasetBatcher(
            self.epoch_iter,
            self.features,
            self.training_set_metadata,
            batch_size,
            self.size,
            ignore_last,
            augmentation_pipeline=augmentation_pipeline,
        )

    @lru_cache(1)
    def __len__(self):
        return next(self.epoch_iter).count()

    @property
    def size(self):
        return len(self)


@DeveloperAPI
class RayDatasetBatcher(Batcher):
    def __init__(
        self,
        dataset_epoch_iterator: Iterator[DatasetPipeline],
        features: Dict[str, Dict],
        training_set_metadata: TrainingSetMetadataDict,
        batch_size: int,
        samples_per_epoch: int,
        ignore_last: bool = False,
        # TODO: figure out correct typing for augmentation_pipeline after refactoring is done
        augmentation_pipeline=None,
    ):
        self.dataset_epoch_iterator = dataset_epoch_iterator
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.training_set_metadata = training_set_metadata
        self.ignore_last = ignore_last
        self.augmentation_pipeline = augmentation_pipeline

        self.features = features
        self.columns = list(features.keys())
        self._sample_feature_name = self.columns[0]
        self.reshape_map = {
            proc_column: training_set_metadata[feature[NAME]].get("reshape")
            for proc_column, feature in features.items()
        }

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
        pipeline = next(self.dataset_epoch_iterator)

        read_parallelism = 1
        if read_parallelism == 1:
            self.dataset_batch_iter = self._create_async_reader(pipeline)
        elif read_parallelism > 1:
            # TODO: consider removing this. doesn't work currently and read performance seems generally
            #  very good with 1 parallelism
            self.dataset_batch_iter = self._create_async_parallel_reader(pipeline, read_parallelism)
        else:
            self.dataset_batch_iter = self._create_sync_reader(pipeline)

        self._step = 0
        self._fetch_next_batch()

    def _fetch_next_batch(self):
        if self.dataset_batch_iter is None:
            self._last_batch = True
            return

        self._last_batch = False
        try:
            self._next_batch = next(self.dataset_batch_iter)
            # If the batch has only one row and self.ignore_last, skip the batch
            # to prevent batchnorm / dropout related Torch errors
            if self.ignore_last and len(self._next_batch[self._sample_feature_name]) == 1:
                raise StopIteration
        except StopIteration:
            self._last_batch = True

    def _prepare_batch(self, batch: pd.DataFrame) -> Dict[str, np.ndarray]:
        res = {}
        for c in self.columns:
            if self.features[c][TYPE] not in _SCALAR_TYPES:
                # Ensure columns stacked instead of turned into np.array([np.array, ...], dtype=object) objects
                res[c] = np.stack(batch[c].values)
            else:
                res[c] = batch[c].to_numpy()

        for c in self.columns:
            reshape = self.reshape_map.get(c)
            if reshape is not None:
                res[c] = res[c].reshape((-1, *reshape))
        return res

    def _augment_batch_fn(self):
        augmentation_pipeline = self.augmentation_pipeline

        def augment_batch(df: pd.DataFrame) -> pd.DataFrame:
            # df is pandas dataframe, where each column is Series, to use data as arrays
            # convert dataframe to dict of arrays
            dict_of_arrays = to_numpy_dataset(df)

            if augmentation_pipeline:
                for c, augmentations in augmentation_pipeline.items():
                    # TODO: convert to debug message when done with development
                    logger.info(f"RayDatasetBatcher applying augmentation pipeline to batch for feature {c}")

                    # apply augmentation pipeline operations to the batch of np.array
                    dict_of_arrays[c] = augmentations(torch.tensor(dict_of_arrays[c])).numpy()

            # convert dict of arrays back to dataframe
            df = from_numpy_dataset(dict_of_arrays)
            return df

        return augment_batch

    def _create_sync_reader(self, pipeline: DatasetPipeline):
        def sync_read():
            for batch in pipeline.iter_batches(prefetch_blocks=0, batch_size=self.batch_size, batch_format="pandas"):
                yield self._prepare_batch(batch)

        return sync_read()

    def _create_async_reader(self, pipeline: DatasetPipeline):
        q = queue.Queue(maxsize=100)
        batch_size = self.batch_size
        augment_batch = self._augment_batch_fn()

        def producer():
            nonlocal pipeline

            try:
                # if augmentation is specified, setup prefetching batch of data
                if self.augmentation_pipeline:
                    pipeline = pipeline.map_batches(augment_batch, batch_size=batch_size, batch_format="pandas")

                for batch in pipeline.iter_batches(prefetch_blocks=0, batch_size=batch_size, batch_format="pandas"):
                    res = self._prepare_batch(batch)
                    q.put(res)
                q.put(None)
            except Exception as e:
                # Ensure any exceptions raised in this background thread are raised on the main thread
                q.put(e)

        def async_read():
            t = threading.Thread(target=producer)
            t.start()
            while True:
                batch = q.get(block=True)
                if isinstance(batch, Exception):
                    # Raise any exceptions from the producer thread
                    raise batch
                if batch is None:
                    break
                yield batch
            t.join()

        return async_read()

    def _create_async_parallel_reader(self, pipeline: DatasetPipeline, num_threads: int):
        q = queue.Queue(maxsize=100)

        batch_size = self.batch_size

        splits = pipeline.split(n=num_threads)

        def producer(i):
            for batch in splits[i].iter_batches(prefetch_blocks=0, batch_size=batch_size, batch_format="pandas"):
                res = self._prepare_batch(batch)
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
