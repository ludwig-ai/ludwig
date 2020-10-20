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

import pandas as pd


class CSVLoadMixin:
    """Reads a CSV file into a Pandas DataFrame."""

    config: dict
    processed_dataset_path: str

    def load_processed_dataset(self) -> pd.DataFrame:
        """Loads the processed CSV into a dataframe.

        :returns: A pandas dataframe
        """
        dataset_csv = os.path.join(self.processed_dataset_path, self.csv_filename)
        return pd.read_csv(dataset_csv)

    @property
    def csv_filename(self):
        return self.config["csv_filename"]
