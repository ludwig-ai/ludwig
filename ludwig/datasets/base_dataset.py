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
import os
from pathlib import Path
import yaml
import pandas as pd


"""An abstract base class that defines a set of methods for download,
preprocess and plug data into the ludwig training API"""


class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(self, dataset_name):
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path, "./text/dataset_config.yaml")
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self.default_cache_dir = self._config_file_contents["cache_location"]
        if self.default_cache_dir is None:
            self._cache_dir = str(Path.home().joinpath('.ludwig_cache'))
        else:
            self._cache_dir = self.default_cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        self._cur_version = self._config_file_contents[dataset_name]
        self._download_path = str(Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_" + str(self._cur_version)
                                                                                     + "/raw.csv"))
        self.processed_data_path = str(Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_" + str(self._cur_version)
                                                                                     + "/processed.csv"))
        self.raw_data_path=""
        self._result_dict = {}


    """Download the raw data to the ludwig cache in the format ~/.ludwig_cache/id
       where is is represented by the name.version of the dataset
       :param dataset_name: (str) the name of the dataset we need to retrieve.
       Returns:
          None
    """
    @abc.abstractmethod
    def download(self, dataset_name) -> None:
        raise NotImplementedError("You will need to implement the download method to download the training data")

    """Process the dataset to get it ready to be plugged into a dataframe
       in the manner needed by the ludwig training API
       Returns:
           None
    """
    @abc.abstractmethod
    def process(self) -> None:
        raise NotImplementedError("You will need to implement the method to process the training data")

    """A pre-op check to see if the raw or processed data exists before
    we perform the next operation , in the case of doing the transforms
    we check the raw data and in the case of the load method we check for
    the existence of the processed data.
    Arguments:
       cur_version (str): the version of the data that we should have already processed
       data_name (str): the name of the dataset that we've read from config
    Returns: True or false whether we can start the loading into a dataframe"""
    def check_file_existence(self, cur_version, data_name, file_name) -> bool:
        if file_name == "raw.csv":
            self.raw_data_path = Path.home().joinpath('.ludwig_cache') \
                .joinpath(data_name + "_" + str(cur_version)).joinpath(file_name)
            final_data_path = self.raw_data_path
        elif file_name == "processed.csv":
            self.processed_data_path = Path.home().joinpath('.ludwig_cache') \
                .joinpath(data_name + "_" + str(cur_version)).joinpath(file_name)
            final_data_path = self.processed_data_path
        return os.path.isfile(final_data_path)
