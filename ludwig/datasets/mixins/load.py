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

import os
from typing import Tuple, Union

import pandas as pd

from ludwig.constants import SPLIT


def _split(data_df, split):
    if SPLIT in data_df:
        data_df[SPLIT] = pd.to_numeric(data_df[SPLIT])
    if split:
        if SPLIT in data_df:
            training_set = data_df[data_df[SPLIT] == 0].drop(columns=[SPLIT])
            val_set = data_df[data_df[SPLIT] == 1].drop(columns=[SPLIT])
            test_set = data_df[data_df[SPLIT] == 2].drop(columns=[SPLIT])
            return training_set, test_set, val_set
        else:
            raise ValueError("The dataset does not have splits, " "load with `split=False`")
    return data_df


class CSVLoadMixin:
    """Reads a CSV file into a Pandas DataFrame."""

    config: dict
    processed_dataset_path: str

    def load_processed_dataset(self, split) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Loads the processed CSV into a dataframe.

        :param split: Splits along 'split' column if present.
        :returns: The preprocessed dataset, or a tuple of (train, validation, test) datasets.
        """
        data_df = pd.read_csv(self.dataset_path)
        return _split(data_df, split)

    @property
    def csv_filename(self):
        return self.config["csv_filename"]

    @property
    def dataset_path(self):
        return os.path.join(self.processed_dataset_path, self.csv_filename)


class ParquetLoadMixin:
    """Reads a Parquet file into a Pandas DataFrame."""

    config: dict
    processed_dataset_path: str

    def load_processed_dataset(self, split) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Loads the processed Parquet into a dataframe.

        :param split: Splits along 'split' column if present
        :returns: The preprocessed dataset, or a tuple of (train, validation, test) datasets.
        """
        dataset_path = os.path.join(self.processed_dataset_path, self.parquet_filename)
        data_df = pd.read_parquet(dataset_path)
        return _split(data_df, split)

    @property
    def parquet_filename(self):
        return self.config["parquet_filename"]
