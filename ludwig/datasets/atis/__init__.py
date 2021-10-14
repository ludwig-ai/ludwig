import os

import pandas as pd
from ludwig.datasets.base_dataset import DEFAULT_CACHE_LOCATION, BaseDataset
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Atis(cache_dir=cache_dir)
    return dataset.load(split=split)


class Atis(KaggleDownloadMixin, CSVLoadMixin, BaseDataset):
    def __init__(
            self,
            cache_dir=DEFAULT_CACHE_LOCATION,
            kaggle_username=None,
            kaggle_key=None
    ):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        super().__init__(dataset_name='atis', cache_dir=cache_dir)

    def 