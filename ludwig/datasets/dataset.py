#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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
import logging
import os
import urllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import tqdm

from ludwig.datasets.archives import extract_archive, is_archive
from ludwig.datasets.kaggle import download_kaggle_dataset

logger = logging.getLogger(__name__)

DEFAULT_CACHE_LOCATION = str(Path.home().joinpath(".ludwig_cache"))


class TqdmUpTo(tqdm):
    """Provides progress bar for `urlretrieve`.

    Taken from: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


@dataclass
class DatasetConfig:
    # The version of the dataset.
    version: str

    # The name of the dataset. Make this a valid python module name, should not contain spaces or dashes.
    name: str

    # The readable description of the dataset
    description: str = ""

    # The kaggle competition this dataset belongs to, or None if this dataset is not hosted by a Kaggle competition.
    kaggle_competition: Optional[str] = None

    # The kaggle dataset ID, or None if this dataset if not hosted by Kaggle.
    kaggle_dataset_id: Optional[str] = None

    # The list of URLs to download.
    download_urls: List[str] = field(default_factory=list)

    # The list of file archives which will be downloaded. If download_urls contains a filename with extension, for
    # example https://domain.com/archive.zip, then archive_filenames does not need to be specified.
    archive_filenames: List[str] = field(default_factory=list)

    # The type of archive (see archives.py). If None archive type will be inferred from the file.
    archive_type: Optional[str] = None

    # The names of files in the dataset (after extraction). Glob-style patterns are supported, see:
    # https://docs.python.org/3/library/glob.html
    dataset_filenames: List[str] = field(default_factory=list)

    # If the dataset contains separate files for training, testing, or validation. Glob-style patterns are supported,
    # see https://docs.python.org/3/library/glob.html
    train_filenames: List[str] = field(default_factory=list)
    validation_filenames: List[str] = field(default_factory=list)
    test_filenames: List[str] = field(default_factory=list)

    # List of column names, for datasets which do not have column names. If specified, will override the column names
    # already present in the dataset.
    columns: List[str] = field(default_factory=list)

    # Custom dataset implementation, must be provided by @register_dataset. See datasets/registry.py
    custom_implementation: Optional[str] = None


class DatasetState(int, Enum):
    """The state of the dataset."""

    NOT_LOADED = 0
    DOWNLOADED = 1
    EXTRACTED = 2
    TRANSFORMED = 3


class Dataset:
    """Base class that defines the public interface for the ludwig dataset API.

    Clients will typically call load(), which processes the dataset according to the config.

    A dataset is processed in 3 phases:
        1. Download       - The dataset files are downloaded to the cache.
        2. Extract        - The dataset files are extracted from an archive (may be a no-op if data is not archived).
        3. Transform      - The dataset is transformed into a format usable for training and is ready to load.
            a. Transform Files      (Files -> Files)
            b. Load Dataframe       (Files -> DataFrame)
            c. Transform Dataframe  (DataFrame -> DataFrame)
            d. Save Processed       (DataFrame -> File)

    The download and extract phases are run for each URL based on the URL type and file extension. After extraction, the
    full set of downloaded and extracted files are collected and passed as a list to the transform stage.

    The transform phase offers customization points for datasets which require preprocessing before they are usable for
    training.
    """

    def __init__(self, config: DatasetConfig, cache_dir: str = DEFAULT_CACHE_LOCATION):
        self.config = config
        self.cache_dir = cache_dir

    @property
    def name(self):
        """The name of the dataset."""
        return self.config.name

    @property
    def version(self):
        """The version of the dataset."""
        return self.config.version

    @property
    def is_kaggle_dataset(self):
        return self.config.kaggle_dataset_id or self.config.kaggle_competition

    @property
    def download_dir(self):
        """Directory where all dataset artifacts are saved."""
        return os.path.join(self.cache_dir, f"{self.name}_{self.version}")

    @property
    def raw_dataset_dir(self):
        """Save path for raw data downloaded from the web."""
        return os.path.join(self.download_dir, "raw")

    @property
    def processed_dataset_dir(self):
        """Save path for processed data."""
        return os.path.join(self.download_dir, "processed")

    @property
    def processed_dataset_filename(self):
        """Filename for processed data."""
        from ludwig.utils.strings_utils import make_safe_filename

        return f"{make_safe_filename(self.config.name)}.parquet"

    @property
    def processed_dataset_path(self):
        """Save path to processed dataset file."""
        return os.path.join(self.processed_dataset_dir, self.processed_dataset_filename)

    @property
    def processed_temp_dir(self):
        """Save path for processed temp data."""
        return os.path.join(self.download_dir, "_processed")

    @property
    def state(self) -> DatasetState:
        """Dataset state."""
        # If transformed exists in cache:
        if os.path.exists(self.processed_dataset_path):
            return DatasetState.TRANSFORMED
        # if extracted files in cache:
        # return DatasetState.EXTRACTED
        # if downloaded url or archive in cache:
        # return DatasetState.DOWNLOADED

        return DatasetState.NOT_LOADED

    @property
    def download_urls(self) -> List[str]:
        return self.config["download_urls"]

    @property
    def download_filenames(self) -> List[str]:
        return [os.path.basename(urlparse(url).path) for url in self.download_urls]

    def load(self, split=False, kaggle_username=None, kaggle_key=None) -> pd.DataFrame:
        """Loads the dataset, downloaded and processing it if needed.

        Note: This method is also responsible for splitting the data, returning a single dataframe if split=False, and a
        3-tuple of train, val, test if split=True.

        :param split: (bool) splits dataset along 'split' column if present. The split column should always have values
        0: train, 1: validation, 2: test.
        """
        if self.state == DatasetState.NOT_LOADED:
            try:
                self.download(kaggle_username=kaggle_username, kaggle_key=kaggle_key)
                self.state = DatasetState.DOWNLOADED
            except Exception:
                logger.exception("Failed to download dataset")
        if self.state == DatasetState.DOWNLOADED:
            # Extract dataset
            try:
                self.extract()
                self.state = DatasetState.EXTRACTED
            except Exception:
                logger.exception("Failed to extract dataset")
        if self.state == DatasetState.EXTRACTED:
            # Transform dataset
            try:
                self.transform()
                self.state = DatasetState.TRANSFORMED
            except Exception:
                logger.exception("Failed to transform dataset")
        if self.state == DatasetState.TRANSFORMED:
            dataset_df = self.load_transformed_dataset()
            if split:
                return self.split(dataset_df)
            else:
                return dataset_df

    def download(self, kaggle_username=None, kaggle_key=None):
        if self.is_kaggle_dataset:
            return download_kaggle_dataset(
                kaggle_dataset_id=self.config.kaggle_dataset_id,
                kaggle_competition=self.config.kaggle_competition,
                kaggle_username=kaggle_username,
                kaggle_key=kaggle_key,
            )
        else:
            for url, filename in zip(self.download_urls, self.download_filenames):
                downloaded_file_path = os.path.join(self.raw_dataset_directory, filename)
                with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                    urllib.request.urlretrieve(url, downloaded_file_path, t.update_to)

    def extract(self) -> List[str]:
        extracted_files = set()
        for download_filename in self.download_filenames:
            download_path = os.path.join(self.raw_dataset_directory, download_filename)
            if is_archive(download_path):
                extracted_files.update(extract_archive(download_path, archive_type=self.config.archive_type))
        return list(extracted_files)

    def transform(self) -> pd.DataFrame:
        pass

    def transform_files(self, file_paths: List[str]) -> List[str]:
        pass

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        pass

    def transform_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Subclasses should override this method if transformation of the dataframe is needed."""
        return dataset

    def save_processed(self, dataset: pd.DataFrame):
        """Saves transformed dataframe as a flat file ludwig can load for training."""
        pass

    def split(self, dataset: pd.DataFrame):
        pass
