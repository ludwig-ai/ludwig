import os
import fsspec
import shutil
from fsspec.core import split_protocol
from ludwig.utils.fs_utils import makedirs
from contextlib import contextmanager
import tempfile
from zipfile import ZipFile


def create_kaggle_client():
    # Need to import here to prevent Kaggle from authenticating on import
    from kaggle import api
    return api


class KaggleDownloadMixin:
    """A mixin to abstract away the details of the kaggle API which includes
    the ability to authenticate against the kaggle API, list the various datasets
    and finally download the dataset, we derive from ZipDownloadMixin to take
    advantage of extracting contents from the archive zip file"""
    config: dict
    raw_dataset_path: str
    raw_temp_path: str
    name: str
    kaggle_username: str
    kaggle_key: str
    is_kaggle_competition: bool

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.  If the user has not specified creds in the
        kaggle.json file we lookup the passed in username and the api key and
        perform authentication.
        """
        with self.update_env(KAGGLE_USERNAME=self.kaggle_username, KAGGLE_KEY=self.kaggle_key):
            # Call authenticate explicitly to pick up new credentials if necessary
            api = create_kaggle_client()
            api.authenticate()

        #os.makedirs(self.raw_temp_path, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.is_kaggle_competition:
                download_func = api.competition_download_files
            else:
                download_func = api.dataset_download_files
            # Download all files for a competition/dataset
            download_func(self.competition_name, path=tmpdir)

            archive_zip = os.path.join(tmpdir, self.archive_filename)
            with ZipFile(archive_zip, 'r') as z:
                z.extractall(tmpdir)

            protocol, _ = split_protocol(self.raw_dataset_path)
            if protocol is not None:
                makedirs(self.raw_dataset_path, exist_ok=True)
                fs = fsspec.filesystem(protocol)
                fs.put(tmpdir, self.raw_dataset_path)
            else:
                shutil.copytree(tmpdir, self.raw_dataset_path)

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
