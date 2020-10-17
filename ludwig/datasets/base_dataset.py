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
import yaml
from pathlib import Path
import pandas as pd


"""A base class that defines the public interface for the ludwig dataset API.
This includes the download, transform and converting the final transformed API
into a resultant dataframe"""

# define a default location for the cache
DEFAULT_CACHE_LOCATION = str(Path.home().joinpath('.ludwig_cache'))


class BaseDataset:

    def __init__(self, dataset_name, cache_location):
        self._dataset_name = dataset_name
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path, "./config/dataset_config.yaml")
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        if cache_location is not None:
            self._cache_location = cache_location
        else:
            self._cache_location = DEFAULT_CACHE_LOCATION
        self._dataset_version = self._config_file_contents[dataset_name]["version"]
        self._download_url = self._config_file_contents[dataset_name]["download_url"]
        self._dataset_file_name = self._config_file_contents[dataset_name]["extracted_file_name"]
        self._download_dir = Path.home().joinpath('.ludwig_cache').joinpath(self._dataset_name + "_"
                                                                         + str(self._dataset_version))
        self._raw_file_name = Path.home().joinpath('.ludwig_cache').joinpath(self._dataset_name + "_"
                                                                    + str(self._dataset_version)). \
            joinpath('raw.csv')
        self._processed_file_name = Path.home().joinpath('.ludwig_cache').joinpath(self._dataset_name + "_"
                                                                         + str(self._dataset_version)). \
            joinpath('processed.csv')
        self._result_dict = {}

    """Download the file from config url that represents the raw unprocessed training data.
       The workflow for this involves unzipping the file and renaming it to raw.csv, which means
       keep trying to download the file till successful.
    :arg:
        None
    :return
        None"""
    def download(self) -> None:
        self.downloaded_raw_dataset()

    """A helper method to verify the download
    :arg
        None
    :return
        True or false identifying whether the file has been downloaded"""
    def _is_downloaded(self) -> bool:
        return os.path.isfile(self._raw_file_name)

    """Process the dataset to get it ready to be plugged into a dataframe
           in the manner needed by the ludwig training API, to do this we create
           a new dictionary that contains the KV pairs in the format that we need.
           If we fail we redownload the file
           Returns:
               None
        """
    def process(self) -> None:
        if not self._is_downloaded():
            self.download()
        self.transform_downloaded_dataset()

    """Now that the ohsumed data is processed load and return it as a pandas dataframe
       if we cant load the dataframe redo the whole workflow
           :return
              A pandas DataFrame
        """
    def load(self) -> pd.DataFrame:
        return self.transform_processed_data_to_dataframe()


