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
import abc
from pathlib import Path
import pandas as pd

# define a default location for the cache
DEFAULT_CACHE_LOCATION = str(Path.home().joinpath('.ludwig_cache'))


class BaseDataset:
    """A base class that defines the public interface for the ludwig dataset API.
    This includes the download, transform and converting the final transformed API
    into a resultant dataframe"""

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

    """Download the file from config url that represents the raw unprocessed training data.
       The workflow for this involves unzipping the file and renaming it to raw.csv, which means
       keep trying to download the file till successful"""
    def download(self) -> None:
        self.download_raw_dataset()

    """A helper method to verify the download
    :returns: True or false identifying whether the file has been downloaded"""
    def is_downloaded(self) -> bool:
        return os.path.isfile(Path.home().joinpath('.ludwig_cache').joinpath(self._dataset_name + "_"
                                                                             + str(self._dataset_version)). \
            joinpath('raw.csv'))

    """A helper method to verify that the processed file exists
        :returns: True or false identifying whether the processed file exists"""
    def is_processed(self) -> bool:
        return os.path.isfile(Path.home().joinpath('.ludwig_cache').joinpath(self._dataset_name + "_"
                                                                             + str(self._dataset_version)). \
            joinpath('processed.csv'))

    """Process the dataset to get it ready to be plugged into a dataframe
           in the manner needed by the ludwig training API, to do this we create
           a new dictionary that contains the KV pairs in the format that we need.
           If we fail we redownload the file"""
    def process(self) -> None:
        if not self.is_downloaded():
            self.download()
        self.process_downloaded_dataset()

    @abc.abstractmethod
    def download_raw_dataset(self):
        raise NotImplementedError("This method needs to exist in the mixins")

    @abc.abstractmethod
    def process_downloaded_dataset(self):
        raise NotImplementedError("This method needs to exist in the mixins")

    @abc.abstractmethod
    def load_processed_dataset(self):
        raise NotImplementedError("This method needs to exist in the mixins")

    """Now that the ohsumed data is processed load and return it as a pandas dataframe
       if we cant load the dataframe redo the whole workflow
        :returns: A pandas DataFrame
    """
    def load(self) -> pd.DataFrame:
        if not self.is_processed():
            self.process()
        return self.load_processed_dataset()


