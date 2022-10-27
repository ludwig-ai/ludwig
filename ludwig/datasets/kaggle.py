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
    filenames: Optional[list] = None,
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
        dataset_or_competition = kaggle_competition or kaggle_dataset_id
        if filenames:
            download_fn = api.competition_download_file if kaggle_competition else api.dataset_download_file
            for filename in filenames:
                download_fn(dataset_or_competition, filename, path=tmpdir)
        else:
            download_fn = api.competition_download_files if kaggle_competition else api.dataset_download_files
            download_fn(dataset_or_competition, path=tmpdir)
    return [os.path.join(download_directory, f) for f in os.listdir(download_directory)]
