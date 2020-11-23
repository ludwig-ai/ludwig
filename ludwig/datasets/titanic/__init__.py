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
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleMixin
from ludwig.datasets.mixins.load import CSVLoadMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Titanic(cache_dir=cache_dir)
    return dataset.load(split=split)


class Titanic(CSVLoadMixin, KaggleMixin, BaseDataset):
    """The Titanic dataset.

    This pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming training data into a destination
    dataframe that can fit into Ludwig's training API.
    """

    config: dict
    raw_temp_path: str
    raw_dataset_path: str
    processed_temp_path: str
    processed_dataset_path: str

    def __init__(self, dataset_name="titanic", cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name, cache_dir)

    def process_downloaded_dataset(self):
        """ The final method where we create a training and test file by iterating through
        both of these files"""
        train_df = pd.read_csv(os.path.join(self.processed_temp_path, "titanic_train.csv"))
        test_df = pd.read_csv(os.path.join(self.processed_temp_path, "titanic_test.csv"))
        train_df["split"] = 0
        test_df["split"] = 2
        final_df = pd.concat([train_df, train_df], axis=1)
        final_df.to_csv(self.processed_dataset_path)

    @property
    def download_url(self):
        return self.config["download_url"]
