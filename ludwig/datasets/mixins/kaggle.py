import os
from contextlib import contextmanager
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDownloadMixin:
    """A mixin to abstract away the details of the kaggle API which includes
    the ability to authenticate against the kaggle API, list the various datasets
    and finally download the dataset"""
    config: dict
    raw_dataset_path: str
    raw_temp_path: str
    name: str
    kaggle_username: str
    kaggle_api_key: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.  If the user has not specified creds in the
        kaggle.json file we lookup the passed in username and the api key and
        perform authentication.
        """
        with self.update_env(KAGGLE_USERNAME=self.kaggle_username, KAGGLE_API_KEY=self.kaggle_api_key):
            api = KaggleApi()
            api.authenticate()
        os.makedirs(self.raw_temp_path, exist_ok=True)
        # Download all files for a competition
        # Signature: competition_download_files(competition, path=None, force=False, quiet=True)
        api.competition_download_files('titanic', path=self.raw_temp_path)
        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @contextmanager
    def update_env(**kwargs):
        override_env = {k: v for k, v in kwargs.items() if v is not None}
        old = os.environ
        try:
            os.environ = {os.environ, override_env}
            yield
        finally:
            os.environ = old

    @property
    def competition_name(self):
        return self.config["competition"]

