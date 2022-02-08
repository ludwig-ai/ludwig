import logging

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.print_utils import print_boxed


def load(cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
    print_boxed("LOADING DATA")
    logging.info("Loading WNMT15 data from Kaggle (this may take a while).")
    dataset = WMT15(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    loaded_dataset = dataset.load(split=False)
    logging.info("Finished loading.")
    return loaded_dataset


@register_dataset(name="wmt15")
class WMT15(CSVLoadMixin, IdentityProcessMixin, KaggleDownloadMixin, BaseDataset):
    """French/English parallel texts for training translation models. Over 22.5 million sentences in French and
    English.

    Additional details:

    https://www.kaggle.com/dhruvildave/en-fr-translation-dataset
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = False
        super().__init__(dataset_name="wmt15", cache_dir=cache_dir)
