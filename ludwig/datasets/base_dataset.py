#! /usr/bin/env python
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

import pandas as pd
import yaml

DEFAULT_CACHE_LOCATION = str(Path.home().joinpath(".ludwig_cache"))
PATH_HERE = os.path.abspath(os.path.dirname(__file__))


def read_config(dataset_name):
    config_path = os.path.join(PATH_HERE, f"{dataset_name}/config.yaml")
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


class BaseDataset:
    """Base class that defines the public interface for the ludwig dataset API.

    This includes the download, transform, and converting the final transformed API into a resultant dataframe.
    """

    def __init__(self, dataset_name, cache_dir):
        self.name = dataset_name
        self.cache_dir = cache_dir

        self.config = read_config(dataset_name)
        self.version = self.config["version"]

    def download(self) -> None:
        """Download the file from config.download_urls and save the file(s) at self.raw_dataset_path."""
        self.download_raw_dataset()

    def process(self) -> None:
        """Process the dataset into a dataframe and save it at self.processed_dataset_path."""
        if not self.is_downloaded():
            self.download()
        self.process_downloaded_dataset()

    def load(self, split=False) -> pd.DataFrame:
        """Loads the processed data from processed_dataset_path into a Pandas DataFrame in memory.

        Note: This method is also responsible for splitting the data, returning a single dataframe if split=False, and a
        3-tuple of train, val, test if split=True.

        :param split: (bool) splits dataset along 'split' column if present. The split column should always have values
        0: train, 1: validation, 2: test.
        """
        if not self.is_processed():
            self.process()
        return self.load_processed_dataset(split)

    @property
    def raw_dataset_path(self):
        """Save path for raw data downloaded from the web."""
        return os.path.join(self.download_dir, "raw")

    @property
    def raw_temp_path(self):
        """Save path for temp data downloaded from the web."""
        return os.path.join(self.download_dir, "_raw")

    @property
    def processed_dataset_path(self):
        """Save path for processed data."""
        return os.path.join(self.download_dir, "processed")

    @property
    def processed_temp_path(self):
        """Save path for processed temp data."""
        return os.path.join(self.download_dir, "_processed")

    @property
    def download_dir(self):
        """Directory where all dataset artifacts are saved."""
        return os.path.join(self.cache_dir, f"{self.name}_{self.version}")

    @abc.abstractmethod
    def download_raw_dataset(self):
        """Download the file from config.download_urls and save the file(s) at self.raw_dataset_path."""
        raise NotImplementedError()

    @abc.abstractmethod
    def process_downloaded_dataset(self):
        """Process the dataset into a dataframe and save it at self.processed_dataset_path."""
        raise NotImplementedError()

    @abc.abstractmethod
    def load_processed_dataset(self, split: bool):
        """Loads the processed data from processed_dataset_path into a Pandas DataFrame in memory.

        Note: This method is also responsible for splitting the data, returning a single dataframe if split=False, and a
        3-tuple of train, val, test if split=True.

        :param split: (bool) splits dataset along 'split' column if present. The split column should always have values
        0: train, 1: validation, 2: test.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_downloaded(self) -> bool:
        """A helper method to verify the download.

        :returns: True or false identifying whether the file has been downloaded
        """
        return self.is_processed() or os.path.exists(self.raw_dataset_path)

    @abc.abstractmethod
    def is_processed(self) -> bool:
        """A helper method to verify that the processed file exists.

        :returns: True or false identifying whether the processed file exists
        """
        return os.path.exists(self.processed_dataset_path)
