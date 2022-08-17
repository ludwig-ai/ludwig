import os

import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_key=None):
    dataset = AmazonEmployeeAccessChallenge(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    return dataset.load(split=split)


@register_dataset(name="amazon_employee_access_challenge")
class AmazonEmployeeAccessChallenge(CSVLoadMixin, KaggleDownloadMixin, BaseDataset):
    """Allstate Claims Severity.

    https://www.kaggle.com/competitions/amazon-employee-access-challenge/data
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = True
        super().__init__(dataset_name="amazon_employee_access_challenge", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        """The final method where we create a concatenated CSV file with both training ant dest data."""
        train_file = self.config["split_filenames"]["train_file"]
        test_file = self.config["split_filenames"]["test_file"]

        train_df = pd.read_csv(os.path.join(self.raw_dataset_path, train_file))
        test_df = pd.read_csv(os.path.join(self.raw_dataset_path, test_file))

        train_df["split"] = 0
        test_df["split"] = 2

        df = pd.concat([train_df, test_df])

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_csv(os.path.join(self.processed_temp_path, self.csv_filename), index=False)
        rename(self.processed_temp_path, self.processed_dataset_path)
