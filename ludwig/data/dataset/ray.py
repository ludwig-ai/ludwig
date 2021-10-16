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
from functools import lru_cache
from typing import Dict, List, Optional, Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa

# TODO(travis): remove import of this from top-level api due to optional deps
from dask.dataframe.extensions import make_array_nonempty
import ray
from ray.data import from_dask
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.extensions import TensorArray, TensorDtype

import tensorflow as tf

from ludwig.data.batcher.base import Batcher
from ludwig.data.dataset.base import Dataset
from ludwig.features.base_feature import InputFeature, OutputFeature
from ludwig.utils.data_utils import DATA_TRAIN_HDF5_FP, to_numpy_dataset
from ludwig.utils.misc_utils import get_proc_features
from ludwig.utils.types import DataFrame


@make_array_nonempty.register(TensorDtype)
def _(dtype):
    return TensorArray._from_sequence([0, np.nan], dtype=dtype)


class RayDataset(object):
    """ Wrapper around ray.data.Dataset. """

    def __init__(self, df: DataFrame, features: Dict[str, Dict], data_hdf5_fp: Optional[str]):
        # for proc_column in features.keys():
        #     df[proc_column] = df[proc_column].astype(TensorDtype())

        self.ds = from_dask(df)
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp

        # def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
        #     npds = to_numpy_dataset(df)
        #     for c in features.keys():
        #         df[c] = TensorArray(npds[c])
        #     return df
        #     # columns = [c for c in df.columns if c in features]
        #     # return pd.DataFrame(
        #     #     columns=columns,
        #     #     data=[TensorArray(npds[c]) for c in columns]
        #     # )

        # def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
        #     for c in features.keys():
        #         df[c] = df[c].astype(TensorDtype())
        #     return df
        #
        # print(f"!!! BEFORE SCHEMA: {self.ds.schema()}")
        # self.ds = self.ds.map_batches(to_tensors, batch_format="pandas")
        # print(f"!!! AFTER SCHEMA: {self.ds.schema()}")

    def pipeline(self, shuffle=True) -> DatasetPipeline:
        pipe = self.ds.repeat()
        if shuffle:
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


# def prepare_dataset_shard(dataset_shard: tf.data.Dataset):
#     # Disable Tensorflow autosharding since the dataset has already been
#     # sharded.
#     options = tf.data.Options()
#     options.experimental_distribute.auto_shard_policy = \
#         tf.data.experimental.AutoShardPolicy.OFF
#     dataset = dataset_shard.with_options(options)
#     return dataset


# def to_tf(
#         dataset: ray.data.Dataset,
#         columns: List[str],
#         output_signature: Dict[str, "tf.TypeSpec"],
#         prefetch_blocks: int = 0,
#         batch_size: int = 1
# ) -> "tf.data.Dataset":
#
#     def make_generator():
#         for batch in dataset.iter_batches(
#                 prefetch_blocks=prefetch_blocks,
#                 batch_size=batch_size,
#                 batch_format="pandas"
#         ):
#             arrays = to_numpy_dataset(batch)
#             yield {
#                 c: arrays[c] for c in columns
#             }
#
#     tf_dataset = tf.data.Dataset.from_generator(
#         make_generator,
#         output_signature=output_signature
#     ).prefetch(tf.data.experimental.AUTOTUNE)
#
#     return tf_dataset


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

        input_features_signature = {
            feature.proc_column: tf.TensorSpec(
                shape=(None, *feature.get_input_shape()),
                dtype=feature.get_input_dtype(),
            )
            for feature in input_features.values()
        }

        output_features_signature = {
            feature.proc_column: tf.TensorSpec(
                shape=(None, *feature.get_output_shape()),
                dtype=feature.get_output_dtype(),
            )
            for feature in output_features.values()
        }

        self.output_signature = {
            **input_features_signature,
            **output_features_signature,
        }

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

    def set_epoch(self, epoch):
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
        # self.dataset_batch_iter = iter(prepare_dataset_shard(
        #     to_tf(
        #         dataset,
        #         columns=self.columns,
        #         output_signature=self.output_signature,
        #         batch_size=self.batch_size,
        #     )
        # ))

        # def gen_batches():
        #     for batch in dataset.iter_batches(
        #             prefetch_blocks=0,
        #             batch_size=self.batch_size,
        #             batch_format="pandas"
        #     ):
        #         arrays = to_numpy_dataset(batch)
        #         yield {
        #             c: arrays[c] for c in self.columns
        #         }
        #
        # self.dataset_batch_iter = gen_batches()

        columns = self.columns

        # def to_tensors(block: pa.Table) -> pa.Table:
        #     npds = to_numpy_dataset(df)
        #     block = block.to_pandas()
        #     block["two"] = TensorArray([pickle.loads(a) for a in block["two"]])
        #     return pa.Table.from_pandas(block)
        #     pa.Table()
        #
        #     # row = []
        #     # for c in columns:
        #     #     print(f"!!! {type(npds[c])} {npds[c]}")
        #     #     t = TensorArray(npds[c])
        #     #     row.append(t)
        #     # return pd.DataFrame(
        #     #     columns=columns,
        #     #     data=[row],
        #     # )
        #     # for c in columns:
        #     #     df[c] = TensorArray(npds[c])
        #     # return df

        # def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
        #     npds = to_numpy_dataset(df)
        #     return pd.DataFrame(
        #         columns=columns,
        #         data=[
        #             [TensorArray(npds[c]) for c in columns]
        #         ],
        #     )
        #
        # self.dataset_batch_iter = dataset.map_batches(
        #     to_tensors, batch_format='pandas', batch_size=self.batch_size
        # ).iter_rows()

        # def gen_batches():
        #     for batch in dataset.map_batches(
        #         to_tensors, batch_format="pandas"
        #     ).iter_batches(
        #             prefetch_blocks=0,
        #             batch_size=self.batch_size,
        #             batch_format="pandas"
        #     ):
        #         yield {
        #             c: batch[c] for c in self.columns
        #         }

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = df[c].astype(TensorDtype())
            return df

        self.dataset_batch_iter = dataset.map_batches(
            to_tensors, batch_format="pandas"
        ).iter_batches(
            prefetch_blocks=0,
            batch_size=self.batch_size,
            batch_format="pandas"
        )

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
