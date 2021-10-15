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
from typing import Dict, List, Optional, Any

# TODO(travis): remove import of this from top-level api due to optional deps
import ray
from ray.data import from_dask
from ray.data.dataset_pipeline import DatasetPipeline

import tensorflow as tf

from ludwig.data.batcher.base import Batcher
from ludwig.data.dataset.base import Dataset
from ludwig.features.base_feature import InputFeature, OutputFeature
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP
from ludwig.utils.misc_utils import get_combined_features, get_proc_features
from ludwig.utils.types import DataFrame


class RayDataset(object):
    """ Wrapper around ray.data.Dataset. """

    def __init__(self, df: DataFrame, features: List[Dict], data_hdf5_fp: Optional[str]):
        self.ds = from_dask(df)
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp

    def pipeline(self) -> DatasetPipeline:
        return self.ds.repeat().random_shuffle()

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

    def create_inference_dataset(
            self,
            dataset: DataFrame,
            tag: str,
            config: Dict[str, Any],
            training_set_metadata: Dict[str, Any]
    ):
        return RayDataset(
            df=dataset,
            features=get_proc_features(config),
            data_hdf5_fp=training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        )

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

    @contextlib.contextmanager
    def initialize_batcher(self, batch_size=128,
                           should_shuffle=True,
                           shuffle_buffer_size=None,
                           seed=0,
                           ignore_last=False,
                           horovod=None):
        return RayDatasetBatcher(
            self.dataset_shard,
            self.input_features,
            self.output_features,
            batch_size
        )

    def __len__(self):
        # TODO(travis): find way to avoid calling this, as it's expensive
        return self.dataset_shard.count()

    @property
    def size(self):
        return len(self)


def prepare_dataset_shard(dataset_shard: tf.data.Dataset):
    # Disable Tensorflow autosharding since the dataset has already been
    # sharded.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset_shard.with_options(options)
    return dataset


def to_tf(
        dataset: ray.data.Dataset,
        output_signature: Dict[str, "tf.TypeSpec"],
        prefetch_blocks: int = 0,
        batch_size: int = 1
) -> "tf.data.Dataset":

    def make_generator():
        for batch in dataset.iter_batches(
                prefetch_blocks=prefetch_blocks,
                batch_size=batch_size,
                batch_format="pandas"):
            yield {
                c: batch[c]
                for c in batch.columns
            }

    dataset = tf.data.Dataset.from_generator(
        make_generator,
        output_signature=output_signature
    )

    return dataset


class RayDatasetBatcher(Batcher):
    def __init__(
            self,
            dataset_pipeline: DatasetPipeline,
            input_features: Dict[str, InputFeature],
            output_features: Dict[str, OutputFeature],
            batch_size: int,
    ):
        self.dataset_epoch_iterator = dataset_pipeline.iter_datasets()
        self.batch_size = batch_size

        input_features_signature = {
            name: tf.TensorSpec(
                shape=(None, *feature.get_input_shape()),
                dtype=feature.get_input_dtype(),
            )
            for name, feature in input_features
        }

        output_features_signature = {
            name: tf.TensorSpec(
                shape=(None, *feature.get_output_shape()),
                dtype=feature.get_output_dtype(),
            )
            for name, feature in output_features
        }

        self.output_signature = {
            **input_features_signature,
            **output_features_signature,
        }
        self.dataset_batch_iter = None
        self._next_batch = None
        self._last_batch = False

    def next_batch(self):
        if self.last_batch():
            raise StopIteration()

        batch = self._next_batch
        self._fetch_next_batch()
        return batch

    def last_batch(self):
        return self._last_batch

    def set_epoch(self, epoch):
        dataset = next(self.dataset_epoch_iterator)
        self.dataset_batch_iter = iter(prepare_dataset_shard(
            to_tf(
                dataset,
                output_signature=self.output_signature,
                batch_size=self.batch_size,
            )
        ))
        self._fetch_next_batch()

    def _fetch_next_batch(self):
        self._last_batch = False
        try:
            self._next_batch = next(self.dataset_batch_iter)
        except StopIteration:
            self._last_batch = True
