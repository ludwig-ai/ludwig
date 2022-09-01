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
from enum import Enum
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from ludwig.constants import SPLIT
from ludwig.datasets.archives import extract_archive, is_archive, list_archive
from ludwig.datasets.dataset_config import DatasetConfig
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


def _list_of_strings(list_or_string: Union[str, List[str]]) -> List[str]:
    """Helper function to accept single string or lists in config."""
    return [list_or_string] if isinstance(list_or_string, str) else list_or_string


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
        if os.path.exists(self.processed_dataset_path):
            return DatasetState.TRANSFORMED
        if all([os.path.exists(os.path.join(self.raw_dataset_dir, filename)) for filename in self.download_filenames]):
            archive_filenames = [f for f in self.download_filenames if is_archive(f)]
            if archive_filenames:
                # Check to see if archive has been extracted.
                extracted_files = [
                    f for a in archive_filenames for f in list_archive(os.path.join(self.raw_dataset_dir, a))
                ]
                if all(os.path.exists(os.path.join(self.raw_dataset_dir, ef)) for ef in extracted_files):
                    return DatasetState.EXTRACTED
                else:
                    return DatasetState.DOWNLOADED
            # If none of the dataset download files are archives, skip extraction phase.
            return DatasetState.EXTRACTED
        return DatasetState.NOT_LOADED

    @property
    def download_urls(self) -> List[str]:
        return _list_of_strings(self.config.download_urls)

    @property
    def download_filenames(self) -> List[str]:
        """Filenames for downloaded files inferred from download_urls."""
        if self.config.archive_filenames:
            return _list_of_strings(self.config.archive_filenames)
        return [os.path.basename(urlparse(url).path) for url in self.download_urls]

    # @property
    # def raw_dataset_filenames(self):
    #     """Returns the set of filenames expected to be present after download and extract."""
    #     expected_filenames = set([
    #         *_list_of_strings(self.config.dataset_filenames),
    #         *_list_of_strings(self.config.train_filenames),
    #         *_list_of_strings(self.config.test_filenames),
    #         *_list_of_strings(self.config.validation_filenames)
    #     ])
    #     # If no files are explicitly declared, infer filenames from download_urls.
    #     return expected_filenames

    def description(self):
        return f"{self.config.name} {self.config.version}\n{self.config.description}"

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
            except Exception:
                logger.exception("Failed to download dataset")
        if self.state == DatasetState.DOWNLOADED:
            # Extract dataset
            try:
                self.extract()
            except Exception:
                logger.exception("Failed to extract dataset")
        if self.state == DatasetState.EXTRACTED:
            # Transform dataset
            try:
                self.transform()
            except Exception:
                logger.exception("Failed to transform dataset")
        if self.state == DatasetState.TRANSFORMED:
            dataset_df = self.load_transformed_dataset()
            if split:
                return self.split(dataset_df)
            else:
                return dataset_df

    def download(self, kaggle_username=None, kaggle_key=None):
        if not os.path.exists(self.raw_dataset_dir):
            os.makedirs(self.raw_dataset_dir)
        if self.is_kaggle_dataset:
            return download_kaggle_dataset(
                self.raw_dataset_dir,
                kaggle_dataset_id=self.config.kaggle_dataset_id,
                kaggle_competition=self.config.kaggle_competition,
                kaggle_username=kaggle_username,
                kaggle_key=kaggle_key,
            )
        else:
            for url, filename in zip(self.download_urls, self.download_filenames):
                downloaded_file_path = os.path.join(self.raw_dataset_dir, filename)
                with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                    urllib.request.urlretrieve(url, downloaded_file_path, t.update_to)

    def extract(self) -> List[str]:
        extracted_files = set()
        for download_filename in self.download_filenames:
            download_path = os.path.join(self.raw_dataset_dir, download_filename)
            if is_archive(download_path):
                extracted_files.update(extract_archive(download_path))
        return list(extracted_files)

    def transform(self) -> pd.DataFrame:
        data_filenames = [
            os.path.join(self.raw_dataset_dir, f) for f in os.listdir(self.raw_dataset_dir) if not is_archive(f)
        ]
        transformed_files = self.transform_files(data_filenames)
        unprocessed_dataframe = self.load_unprocessed_dataframe(transformed_files)
        transformed_dataframe = self.transform_dataframe(unprocessed_dataframe)
        self.save_processed(transformed_dataframe)
        pass

    def transform_files(self, file_paths: List[str]) -> List[str]:
        """Transform data files before loading to dataframe.

        Subclasses should override this method to process files before loading dataframe, calling the base class
        implementation first to get the list of data files.
        """
        data_directories = [p for p in file_paths if os.path.isdir(p)]
        data_files = [p for p in file_paths if not os.path.isdir(p)]
        if not os.path.exists(self.processed_dataset_dir):
            os.makedirs(self.processed_dataset_dir)
        # Symlinks any data directories (ex. image directories) into processed directory to avoid unnecessary copy.
        for source_directory in data_directories:
            dest_directory = os.path.join(self.processed_dataset_dir, os.path.basename(source_directory))
            if not os.path.exists(dest_directory):
                os.symlink(source_directory, dest_directory)
        return data_files

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        """Load dataset files into a dataframe."""
        if len(file_paths) == 1:
            path_to_load = file_paths[0]
            file_extension = os.path.splitext(path_to_load)[-1].lower()
            if file_extension == ".parquet":
                return pd.read_parquet(path_to_load)
            elif file_extension == ".csv":
                return pd.read_csv(path_to_load)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Transforms a dataframe of the entire dataset.

        Subclasses should override this method if transformation of the dataframe is needed.
        """
        if self.config.columns:
            dataframe = dataframe.set_axis(self.config.columns, axis=1)
        return dataframe

    def save_processed(self, dataframe: pd.DataFrame):
        """Saves transformed dataframe as a flat file ludwig can load for training."""
        if not os.path.exists(self.processed_dataset_dir):
            os.makedirs(self.processed_dataset_dir)
        dataframe.to_parquet(self.processed_dataset_path)

    def load_transformed_dataset(self):
        """Load processed dataset into a dataframe."""
        return pd.read_parquet(self.processed_dataset_path)

    def split(self, dataset: pd.DataFrame):
        if SPLIT in dataset:
            dataset[SPLIT] = pd.to_numeric(dataset[SPLIT])
            training_set = dataset[dataset[SPLIT] == 0].drop(columns=[SPLIT])
            val_set = dataset[dataset[SPLIT] == 1].drop(columns=[SPLIT])
            test_set = dataset[dataset[SPLIT] == 2].drop(columns=[SPLIT])
            return training_set, test_set, val_set
        else:
            raise ValueError(f"The dataset does not a '{SPLIT}' column, load with `split=False`")
        return dataset
