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

from abc import ABC, abstractmethod

from ludwig.utils.types import DataFrame


class DataFrameEngine(ABC):
    @abstractmethod
    def df_like(self, df, proc_cols):
        raise NotImplementedError()

    @abstractmethod
    def parallelize(self, data):
        raise NotImplementedError()

    @abstractmethod
    def persist(self, data):
        raise NotImplementedError()

    @abstractmethod
    def compute(self, data):
        raise NotImplementedError()

    @abstractmethod
    def from_pandas(self, df):
        raise NotImplementedError()

    @abstractmethod
    def map_objects(self, series, map_fn, meta=None):
        raise NotImplementedError()

    @abstractmethod
    def map_partitions(self, series, map_fn, meta=None):
        raise NotImplementedError()

    @abstractmethod
    def map_batches(self, df, map_fn, enable_tensor_extension_casting=True):
        raise NotImplementedError()

    @abstractmethod
    def apply_objects(self, series, map_fn, meta=None):
        raise NotImplementedError()

    @abstractmethod
    def reduce_objects(self, series, reduce_fn):
        raise NotImplementedError()

    @abstractmethod
    def split(self, df, probabilities):
        """Splits the input DataFrame into sections with the given proportions."""
        raise NotImplementedError()

    @abstractmethod
    def to_parquet(self, df, path, index=False):
        """Write the input DataFrame to the path in the Parquet format.

        Optionally includes the DataFrame index in the Parquet file.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_predictions(self, df: DataFrame, path: str):
        raise NotImplementedError()

    @abstractmethod
    def read_predictions(self, path: str) -> DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def to_ray_dataset(self, df):
        raise NotImplementedError()

    @property
    @abstractmethod
    def array_lib(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def df_lib(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def partitioned(self):
        raise NotImplementedError()

    @abstractmethod
    def set_parallelism(self, parallelism):
        raise NotImplementedError()
