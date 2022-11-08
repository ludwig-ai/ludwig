import os
from contextlib import contextmanager
from typing import Optional

from ludwig.utils.fs_utils import upload_output_directory


def create_kaggle_client():
    # Need to import here to prevent Kaggle from authenticating on import
    from kaggle import api

    return api


@contextmanager
def update_env(**kwargs):
    override_env = {k: v for k, v in kwargs.items() if v is not None}
    old = os.environ.copy()
    try:
        os.environ.update(override_env)
        yield
    finally:
        os.environ = old


def download_kaggle_dataset(
    download_directory: str,
    kaggle_dataset_id: Optional[str] = None,
    kaggle_competition: Optional[str] = None,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
):
    """Download all files in a kaggle dataset. One of kaggle_dataset_id,

    If the user has not specified creds in the kaggle.json file we lookup the passed in username and the api key and
    perform authentication.
    """
    with update_env(KAGGLE_USERNAME=kaggle_username, KAGGLE_KEY=kaggle_key):
        # Call authenticate explicitly to pick up new credentials if necessary
        api = create_kaggle_client()
        api.authenticate()
    with upload_output_directory(download_directory) as (tmpdir, _):
        if kaggle_competition:
            api.competition_download_files(kaggle_competition, path=tmpdir)
        else:
            api.dataset_download_files(kaggle_dataset_id, path=tmpdir)
    return [os.path.join(download_directory, f) for f in os.listdir(download_directory)]
