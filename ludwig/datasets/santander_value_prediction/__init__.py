#! /usr/bin/env python
# Copyright (c) 2021 Uber Technologies, Inc.
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

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin
from ludwig.datasets.registry import register_dataset


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_key=None):
    dataset = SantanderValuePrediction(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    return dataset.load(split=split)


@register_dataset(name="santander_value_prediction")
class SantanderValuePrediction(CSVLoadMixin, IdentityProcessMixin, KaggleDownloadMixin, BaseDataset):
    """The Santander Value Prediction Challenge dataset.

    Additional details:

    https://www.kaggle.com/c/santander-value-prediction-challenge
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = True
        super().__init__(dataset_name="santander_value_prediction", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        super().process_downloaded_dataset()
        processed_df = pd.read_csv(os.path.join(self.processed_dataset_path, self.csv_filename))
        # Ensure feature column names are strings (some are numeric); keep special names as is
        processed_df.columns = ["C" + str(col) for col in processed_df.columns]
        processed_df.rename(columns={"CID": "ID", "Ctarget": "target", "Csplit": "split"}, inplace=True)
        processed_df.to_csv(os.path.join(self.processed_dataset_path, self.csv_filename), index=False)
