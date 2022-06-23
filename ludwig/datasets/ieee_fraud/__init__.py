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
from ludwig.datasets.mixins.process import MultifileJoinProcessMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_key=None):
    dataset = IEEEFraud(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    return dataset.load(split=split)


@register_dataset(name="ieee_fraud")
class IEEEFraud(CSVLoadMixin, MultifileJoinProcessMixin, KaggleDownloadMixin, BaseDataset):
    """The IEEE-CIS Fraud Detection Dataset https://www.kaggle.com/c/ieee-fraud-detection/overview."""

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = True
        super().__init__(dataset_name="ieee_fraud", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        downloaded_files = self.download_filenames
        filetype = self.download_file_type

        train_files = ["train_identity.csv", "train_transaction.csv"]
        test_files = ["test_identity.csv", "test_transaction.csv"]

        train_dfs, test_dfs = {}, {}

        for split_name, filename in downloaded_files.items():
            file_df = self.read_file(filetype, filename, header=0)
            if filename in train_files:
                train_dfs[split_name] = file_df
            elif filename in test_files:
                test_dfs[split_name] = file_df

        # Merge on TransactionID
        final_train = pd.merge(
            train_dfs["train_transaction"], train_dfs["train_identity"], on="TransactionID", how="left"
        )

        makedirs(self.processed_dataset_path)
        # Only save train split as test split has no ground truth labels
        final_train.to_csv(os.path.join(self.processed_dataset_path, self.csv_filename), index=False)
