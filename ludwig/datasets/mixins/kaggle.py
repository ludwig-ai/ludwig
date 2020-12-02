import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DEFAULT_CACHE_LOCATION = str(Path.home().joinpath('.ludwig_cache'))
api = KaggleApi()
api.authenticate()


class KaggleMixin:
    """A mixin to abstract away the details of the kaggle API which includes
    the ability to authenticate against the kaggle API, list the various datasets
    and finally download the dataset"""
    config: dict
    raw_dataset_path: str
    raw_temp_path: str
    name: str

    def list_downloads(self) -> list:
        """In kaggle they use the term competitions, here we list all
        competition objects associated with the titanic data and return that as a list
        :Return:
            a list of competition objects associated with Titanic"""

        return api.competitions_list(search="titanic")

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

    @property
    def download_urls(self):
        return self.config["download_urls"]
