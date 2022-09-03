import os

import numpy as np
import pandas as pd

from ludwig.constants import SPLIT
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True, kaggle_username=None, kaggle_key=None):
    dataset = NoShowAppointments(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    return dataset.load(split=split)


@register_dataset(name="noshow_appointments")
class NoShowAppointments(CSVLoadMixin, KaggleDownloadMixin, BaseDataset):
    """Dataset containing Medical Appointment No Shows."""

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = False
        super().__init__(dataset_name="noshow_appointments", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        """The final method where we create a concatenated CSV file with both training and test data."""
        df = pd.read_csv(os.path.join(self.raw_dataset_path, self.csv_filename))
        df[SPLIT] = np.random.choice(3, len(df), p=(0.7, 0.1, 0.2)).astype(np.int8)

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_csv(os.path.join(self.processed_temp_path, self.csv_filename), index=False)
        rename(self.processed_temp_path, self.processed_dataset_path)
