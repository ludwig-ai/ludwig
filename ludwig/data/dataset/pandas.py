#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas import DataFrame

from ludwig.constants import TRAINING
from ludwig.data.batcher.base import Batcher
from ludwig.data.batcher.random_access import RandomAccessBatcher
from ludwig.data.dataset.base import Dataset, DatasetManager
from ludwig.data.sampler import DistributedSampler
from ludwig.distributed import DistributedStrategy
from ludwig.features.base_feature import BaseFeature
from ludwig.utils.dataframe_utils import from_numpy_dataset, to_numpy_dataset, to_scalar_df
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc_utils import get_proc_features

if TYPE_CHECKING:
    from ludwig.backend.base import Backend

logger = logging.getLogger(__name__)

# Key for storing the path to the training Parquet cache in metadata
DATA_TRAIN_PARQUET_FP = "data_train_parquet_fp"

# Legacy key -- kept for backward-compat loading of old caches
DATA_TRAIN_HDF5_FP = "data_train_hdf5_fp"


def _shapes_path(data_fp):
    """Return the path to the column-shapes sidecar JSON file for a given Parquet cache file."""
    return os.path.splitext(data_fp)[0] + ".shapes.json"


def _save_parquet(data_fp, data):
    """Save a preprocessed dataset (dict of numpy arrays) to Parquet.

    Multi-dimensional columns (e.g. images with shape [H, W, C]) are flattened to 1-D
    before writing because Parquet cannot natively represent N-D arrays inside cells.
    The original shapes are persisted in a sidecar JSON file so that ``_load_parquet``
    can restore them.
    """
    from ludwig.utils.data_utils import save_json

    dataset = data if isinstance(data, dict) else to_numpy_dataset(data)

    column_shapes: dict[str, list[int]] = {}
    flat_dataset: dict[str, np.ndarray] = {}
    for col, arr in dataset.items():
        arr = np.asarray(arr)
        if arr.ndim > 2:
            # Record the per-sample shape (everything after the batch dimension)
            column_shapes[col] = list(arr.shape[1:])
            # Flatten each sample to 1-D so Parquet can store it
            flat_dataset[col] = arr.reshape(arr.shape[0], -1)
        else:
            flat_dataset[col] = arr

    df = from_numpy_dataset(flat_dataset)
    df.to_parquet(data_fp, engine="pyarrow", index=False)

    # Persist shapes sidecar (even if empty, so _load_parquet can always read it)
    save_json(_shapes_path(data_fp), column_shapes)


def _load_parquet(data_fp):
    """Load a preprocessed dataset from Parquet, returning a dict of numpy arrays.

    If a sidecar ``*.shapes.json`` file exists alongside the Parquet file the
    recorded shapes are used to restore multi-dimensional columns.
    """
    from ludwig.utils.data_utils import load_json

    df = pd.read_parquet(data_fp, engine="pyarrow")
    dataset = to_numpy_dataset(df)

    # Restore N-D shapes if available
    shapes_fp = _shapes_path(data_fp)
    if os.path.exists(shapes_fp):
        column_shapes = load_json(shapes_fp)
        for col, shape in column_shapes.items():
            if col in dataset:
                arr = dataset[col]
                dataset[col] = arr.reshape(arr.shape[0], *shape)

    return dataset


def _load_dataset(dataset):
    """Load a dataset from a file path (Parquet or legacy HDF5) or return as-is if already in-memory."""
    if isinstance(dataset, str):
        if dataset.endswith(".parquet"):
            return _load_parquet(dataset)
        elif dataset.endswith(".hdf5") or dataset.endswith(".h5"):
            # Legacy HDF5 loading for backward compatibility
            from ludwig.utils.data_utils import load_hdf5

            logger.info(f"Loading legacy HDF5 cache: {dataset}. Consider re-preprocessing to use Parquet.")
            return to_numpy_dataset(load_hdf5(dataset))
        else:
            raise ValueError(f"Unsupported cache format: {dataset}. Expected .parquet or .hdf5")
    return dataset


class PandasDataset(Dataset):
    def __init__(self, dataset, features, data_cache_fp):
        self.features = features
        self.data_cache_fp = data_cache_fp

        dataset = _load_dataset(dataset)
        self.dataset = to_numpy_dataset(dataset)
        self.size = len(list(self.dataset.values())[0])

    def to_df(self, features: Iterable[BaseFeature] | None = None) -> DataFrame:
        """Convert the dataset to a Pandas DataFrame."""
        if features:
            return from_numpy_dataset({feature.feature_name: self.dataset[feature.proc_column] for feature in features})
        return from_numpy_dataset(self.dataset)

    def to_scalar_df(self, features: Iterable[BaseFeature] | None = None) -> DataFrame:
        return to_scalar_df(self.to_df(features))

    def get(self, proc_column, idx=None):
        if idx is None:
            idx = range(self.size)
        return self.dataset[proc_column][idx]

    def get_dataset(self) -> dict[str, np.ndarray]:
        return self.dataset

    def __len__(self):
        return self.size

    @property
    def processed_data_fp(self) -> str | None:
        return self.data_cache_fp

    @property
    def in_memory_size_bytes(self) -> int:
        df = self.to_df()
        return df.memory_usage(deep=True).sum() if df is not None else 0

    @contextlib.contextmanager
    def initialize_batcher(
        self,
        batch_size: int = 128,
        should_shuffle: bool = True,
        random_seed: int = default_random_seed,
        ignore_last: bool = False,
        distributed: DistributedStrategy = None,
        augmentation_pipeline=None,
    ) -> Batcher:
        sampler = DistributedSampler(
            len(self), shuffle=should_shuffle, random_seed=random_seed, distributed=distributed
        )
        batcher = RandomAccessBatcher(
            self,
            sampler,
            batch_size=batch_size,
            ignore_last=ignore_last,
            augmentation_pipeline=augmentation_pipeline,
        )
        yield batcher


class PandasDatasetManager(DatasetManager):
    def __init__(self, backend: Backend):
        self.backend: Backend = backend

    def create(self, dataset, config, training_set_metadata) -> Dataset:
        cache_fp = training_set_metadata.get(DATA_TRAIN_PARQUET_FP) or training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        return PandasDataset(dataset, get_proc_features(config), cache_fp)

    def save(self, cache_path, dataset, config, training_set_metadata, tag) -> Dataset:
        # Ensure path ends with .parquet
        if not cache_path.endswith(".parquet"):
            cache_path = os.path.splitext(cache_path)[0] + ".parquet"
        _save_parquet(cache_path, dataset)
        if tag == TRAINING:
            training_set_metadata[DATA_TRAIN_PARQUET_FP] = cache_path
        return dataset

    def can_cache(self, skip_save_processed_input) -> bool:
        return self.backend.is_coordinator() and not skip_save_processed_input

    @property
    def data_format(self) -> str:
        return "parquet"
