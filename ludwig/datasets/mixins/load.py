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

import os
from typing import Tuple, Union

import pandas as pd

from ludwig.constants import SPLIT


class CSVLoadMixin:
    """Reads a CSV file into a Pandas DataFrame."""

    config: dict
    processed_dataset_path: str

    def load_processed_dataset(self, split) -> Union[pd.DataFrame,
                                                     Tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.DataFrame]]:
        """Loads the processed CSV into a dataframe.

        :param split: Splits along 'split' column if present
        :returns: A pandas dataframe
        """
        dataset_csv = os.path.join(self.processed_dataset_path,
                                   self.csv_filename)
        data_df = pd.read_csv(dataset_csv)
        if SPLIT in data_df:
            data_df[SPLIT] = pd.to_numeric(data_df[SPLIT])
        if split:
            if SPLIT in data_df:
                training_set = data_df[data_df[SPLIT] == 0]
                val_set = data_df[data_df[SPLIT] == 1]
                test_set = data_df[data_df[SPLIT] == 2]
                return training_set, test_set, val_set
            else:
                raise ValueError("The dataset does not have splits, "
                                 "load with `split=False`")
        return data_df

    @property
    def csv_filename(self):
        return self.config["csv_filename"]

