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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DatasetConfig:
    """The configuration of a Ludwig dataset."""

    # The version of the dataset.
    version: str

    # The name of the dataset. Make this a valid python module name, should not contain spaces or dashes.
    name: str

    # The readable description of the dataset
    description: str = ""

    # Optional. The (suggested) output features for this dataset. Helps users discover new datasets and filter for
    # relevance to a specific machine learning setting.
    output_features: List[dict] = field(default_factory=list)

    # The kaggle competition this dataset belongs to, or None if this dataset is not hosted by a Kaggle competition.
    kaggle_competition: Optional[str] = None

    # The kaggle dataset ID, or None if this dataset if not hosted by Kaggle.
    kaggle_dataset_id: Optional[str] = None

    # The list of URLs to download.
    download_urls: Union[str, List[str]] = field(default_factory=list)

    # The list of file archives which will be downloaded. If download_urls contains a filename with extension, for
    # example https://domain.com/archive.zip, then archive_filenames does not need to be specified.
    archive_filenames: Union[str, List[str]] = field(default_factory=list)

    # The names of files in the dataset (after extraction). Glob-style patterns are supported, see
    # https://docs.python.org/3/library/glob.html
    dataset_filenames: Union[str, List[str]] = field(default_factory=list)

    # If the dataset contains separate files for training, testing, or validation. Glob-style patterns are supported,
    # see https://docs.python.org/3/library/glob.html
    train_filenames: Union[str, List[str]] = field(default_factory=list)
    validation_filenames: Union[str, List[str]] = field(default_factory=list)
    test_filenames: Union[str, List[str]] = field(default_factory=list)

    # If the dataset contains additional referenced files or directories (ex. images or audio) list them here and they
    # will be copied to the same location as the processed dataset. Glob-style patterns are supported,
    # see https://docs.python.org/3/library/glob.html
    preserve_paths: Union[str, List[str]] = field(default_factory=list)

    # Optionally verify integrity of the dataset by providing sha256 checksums for important files. Maps filename to
    # sha256 digest.  Use `sha256sum <filename>` on linux, `shasum -a 256 <filename>` on Mac to get checksums.
    # If verification fails, loading the dataset will fail with a ValueError.
    # If no sha256 digests are in the config, a warning is logged and the dataset will load without verification.
    sha256: Dict[str, str] = field(default_factory=dict)

    # List of column names, for datasets which do not have column names. If specified, will override the column names
    # already present in the dataset.
    columns: List[str] = field(default_factory=list)

    # Optional dictionary which maps column name to column type. Column's will be converted to the requested type, or
    # will be inferred from the dataset by default.
    column_types: Dict[str, str] = field(default_factory=dict)

    # The loader module and class to use, relative to ludwig.datasets.loaders. Only change this if the dataset requires
    # processing which is not handled by the default loader.
    loader: str = "dataset_loader.DatasetLoader"
