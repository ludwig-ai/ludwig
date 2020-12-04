import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


class KaggleDownloadMixin:
    """A mixin to abstract away the details of the kaggle API which includes
    the ability to authenticate against the kaggle API, list the various datasets
    and finally download the dataset"""
    config: dict
    raw_dataset_path: str
    raw_temp_path: str
    name: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.
        """
        os.makedirs(self.raw_temp_path, exist_ok=True)
        # Download all files for a competition
        # Signature: competition_download_files(competition, path=None, force=False, quiet=True)
        api.competition_download_files('titanic', path=self.raw_temp_path)
        os.rename(self.raw_temp_path, self.raw_dataset_path)

