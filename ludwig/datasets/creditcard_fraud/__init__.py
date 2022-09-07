import os

import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_key=None):
    dataset = CreditCardFraud(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    return dataset.load(split=split)


@register_dataset(name="creditcard_fraud")
class CreditCardFraud(CSVLoadMixin, KaggleDownloadMixin, BaseDataset):
    """The Machine Learning Group ULB Dataset https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud."""

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = False
        super().__init__(dataset_name="creditcard_fraud", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        df = pd.read_csv(os.path.join(self.raw_dataset_path, self.csv_filename))

        # Train/Test split like https://www.kaggle.com/competitions/1056lab-fraud-detection-in-credit-card/overview
        df.sort_values(by=["Time"])
        df.loc[:198365, "split"] = 0
        df.loc[198365:, "split"] = 2
        df.split = df.split.astype(int)

        makedirs(self.processed_dataset_path)
        df.to_csv(os.path.join(self.processed_dataset_path, self.csv_filename), index=False)
