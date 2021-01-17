import os
from contextlib import contextmanager
from zipfile import ZipFile


def create_kaggle_client():
    # Need to import here to prevent Kaggle from authenticating on import
    from kaggle import api
    return api


class KaggleDownloadMixin:
    """A mixin to abstract away the details of the kaggle API which includes
    the ability to authenticate against the kaggle API, list the various datasets
    and finally download the dataset, we derive from ZipDownloadMixin to take
    advantage of extracting contents from the titanic zip file"""
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
            # Call authenticate explicitly to pick up new credentials if necessary
            api = create_kaggle_client()
            api.authenticate()
        os.makedirs(self.raw_temp_path, exist_ok=True)

        # Download all files for a competition
        api.competition_download_files(self.competition_name, path=self.raw_temp_path)

        titanic_zip = os.path.join(self.raw_temp_path, self.archive_filename)
        with ZipFile(titanic_zip, 'r') as z:
            z.extractall(self.raw_temp_path)
        os.rename(self.raw_temp_path, self.raw_dataset_path)


    @contextmanager
    def update_env(self, **kwargs):
        override_env = {k: v for k, v in kwargs.items() if v is not None}
        old = os.environ.copy()
        try:
            os.environ.update(override_env)
            yield
        finally:
            os.environ = old

    @property
    def competition_name(self):
        return self.config["competition"]

    @property
    def archive_filename(self):
        return self.config["archive_filename"]

