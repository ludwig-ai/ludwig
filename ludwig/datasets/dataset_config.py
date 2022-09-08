from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DatasetConfig:
    """A config object which declares a Ludwig dataset, making it available in ludwig.datasets."""

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

    # List of column names, for datasets which do not have column names. If specified, will override the column names
    # already present in the dataset.
    columns: List[str] = field(default_factory=list)

    # The loader module and class to use, relative to ludwig.datasets.loaders. Only change this if the dataset requires
    # processing which is not handled by the default loader.
    loader: str = "dataset_loader.DatasetLoader"
