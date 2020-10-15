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
import csv
import os
import shutil
from pathlib import Path
import pandas as pd
from ludwig.datasets.base_dataset import BaseDataset

"""A class to download, process and return the reuters training data to be plugged into the Ludwig training API"""


def load(cache=None):
    dataset = Reuters(cache)
    return dataset.load()


class Reuters(BaseDataset):

    def __init__(self, passed_in_cache_dir):
        super().__init__("reuters")
        if passed_in_cache_dir is not None:
            self._cache_dir = passed_in_cache_dir
        self._source_location = os.path.join(self._initial_path, "./text/reuters/reuters-allcats.csv")
        self._cur_version = self._config_file_contents["reuters"]

    """Download the ohsumed raw data to the ludwig cache in the format ~/.ludwig_cache/id
       where is is represented by the name.version of the dataset
       :param dataset_name: (str) the name of the dataset we need to retrieve.
       Returns: 
          None
    """
    def download(self, dataset_name) -> None:
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("reuters_"
                                                       + str(self._cur_version)).joinpath('raw.csv')
        shutil.copy(self._source_location, download_path)
        result = os.path.isfile(download_path)
        if not result:
            raise FileNotFoundError("The raw data was not downloaded correctly")

    """Process the ohsumed dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API, to do this we create
       a new dictionary that contains the KV pairs in the format that we need.
       Returns:
           None
    """
    def process(self) -> None:
        if super().check_file_existence(self._cur_version, "reuters", "raw.csv"):
            dict_reader = csv.DictReader(open(self._download_path))
            value_to_store = None
            for row in dict_reader:
                for key, value in row.items():
                    if key == "class":
                        value_to_store = value
                    else:
                        key_to_store = value
                        self._result_dict[key_to_store] = value_to_store
        else:
            raise FileNotFoundError("The raw dataset for reuters was not found")
        try:
            with open(self.processed_data_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self._result_dict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

    """Now that the reuters data is processed load and return it as a pandas dataframe
       Returns:
          A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
        column_names = ["text", "class"]
        if super().check_file_existence(self._cur_version, "reuters", "processed.csv"):
            return pd.read_csv(self.processed_data_path, names=column_names)
        else:
            raise FileNotFoundError("The transformed data for reuters does not exist so cant return pandas dataframe")


