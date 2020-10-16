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
import abc
import csv
import os
import yaml
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
from zipfile import ZipFile


"""An abstract base class that defines a set of methods for download,
preprocess and plug data into the ludwig training API"""

# define a default location for the cache
DEFAULT_CACHE_LOCATION = str(Path.home().joinpath('.ludwig_cache'))


class BaseDataset(metaclass=abc.ABCMeta):

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
    @abc.abstractmethod
    def download(self) -> None:
        with urlopen(self._download_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(self._download_dir)
        # we downloaded the file, now check that this file exists
        downloaded_file = self._download_dir.joinpath(self._dataset_file_name)

        # rename the file to raw.csv
        if os.path.isfile(downloaded_file):
            os.rename(downloaded_file, self._raw_file_name)

        # check for file existence and recursively call ourself if we werent successful
        if not os.path.isfile(self._raw_file_name):
            self.download()

    """Process the dataset to get it ready to be plugged into a dataframe
           in the manner needed by the ludwig training API, to do this we create
           a new dictionary that contains the KV pairs in the format that we need.
           If we fail we redownload the file
           Returns:
               None
        """
    @abc.abstractmethod
    def process(self) -> None:
        if self.check_file_existence(self._raw_file_name):
            dict_reader = csv.DictReader(open(self._raw_file_name))
            value_to_store = None
            for row in dict_reader:
                for key, value in row.items():
                    if key == "class":
                        value_to_store = value
                    else:
                        key_to_store = value
                        self._result_dict[key_to_store] = value_to_store
        else:
            self.download()
            self.process()
        try:
            with open(self._processed_file_name, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self._result_dict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

    """Now that the ohsumed data is processed load and return it as a pandas dataframe
       if we cant load the dataframe redo the whole workflow
           :return
              A pandas DataFrame
        """
    def load(self) -> pd.DataFrame:
        column_names = ["text", "class"]
        if self.check_file_existence(self._processed_file_name):
            return pd.read_csv(self._processed_file_name, names=column_names)
        else:
            self.download()
            self.process()
            self.load()

    """A pre-op check to see if the raw or processed file exists as a step to performing
    the next step in the workflow.
    :arg
       file_path (str): the full path to the file to search for
    :return 
        True or false whether we can start the loading into a dataframe"""
    def check_file_existence(self, file_path) -> bool:
        return os.path.isfile(file_path)
