import tempfile
import pandas as pd
import os
import zipfile
from shutil import copy
from unittest import mock

from ludwig.datasets.base_dataset import DEFAULT_CACHE_LOCATION
from ludwig.datasets.titanic import Titanic


class FakeTitanicDataset(Titanic):
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(cache_dir=cache_dir)


def test_download_titanic_dataset(tmpdir):
    titanic_train_df = pd.DataFrame({
        'passenger_id': [1216, 699, 234],
        'pclass': [3, 3, 4],
        'name': ['sai bo', 'bo staff', 'tae kwan nic'],
        'sex': ["female", "male", "male"],
        'age': [38, 28, 18],
        'sibsp': [0,1,0],
        'parch': [1,1,2],
        'ticket': [335432, 315089, 322472],
        'fare': [7.7333, 8.6625, 9.8765],
        'cabin': [1, 2, 4],
        'embarked': ["C", "Q", "S"],
        'boat': [0, 0, 0],
        'body': [0, 1, 0],
        'home.dest': ["Croatia", "Italy","Sweden"],
        'survived': [0, 1, 0]
    })

    titanic_test_df = pd.DataFrame({
        'passenger_id': [1216, 699, 234],
        'pclass': [3, 3, 4],
        'name': ['mo bo', 'bo bo bo', 'Rafael Nadal'],
        'sex': ["female", "male", "male"],
        'age': [28, 18, 30],
        'sibsp': [0, 1, 0],
        'parch': [1, 1, 2],
        'ticket': [335412, 215089, 922472],
        'fare': [17.7333, 18.6625, 19.8765],
        'cabin': [2, 2, 1],
        'embarked': ["Q", "Q", "C"],
        'boat': [0, 0, 0],
        'body': [0, 1, 0],
        'home.dest': ["Sweden", "Slovenia", "Italy"],
        'survived': [0, 1, 0]
    })

    with tempfile.TemporaryDirectory() as source_dir:
        train_fname = os.path.join(source_dir, 'train.csv')
        titanic_train_df.to_csv(train_fname, index=False)

        test_fname = os.path.join(source_dir, 'test.csv')
        titanic_test_df.to_csv(test_fname, index=False)

        archive_filename = os.path.join(source_dir, 'titanic.zip')
        with zipfile.ZipFile(archive_filename, "w") as z:
            z.write(train_fname)
            z.write(test_fname)

        train_outname = os.path.join(tmpdir, 'train.csv')
        test_outname = os.path.join(tmpdir, 'test.csv')

        config = {
            'version': 1.0,
            'competition': 'titanic',
            'archive_filename': 'titanic.zip',
            'split_filenames': {
                'train_file': train_outname,
                'test_file': test_outname,
            },
            'csv_filename': 'titanic.csv',
        }

        def download_files(competition_name, path):
            assert competition_name == 'titanic'
            copy(archive_filename, path)

        with mock.patch('ludwig.datasets.base_dataset.read_config',
                        return_value=config):
            with mock.patch('ludwig.datasets.mixins.kaggle.create_kaggle_client') as mock_kaggle_cls:
                mock_kaggle_api = mock.MagicMock()
                mock_kaggle_api.competition_download_files = download_files
                mock_kaggle_cls.return_value = mock_kaggle_api

                titanic_handle = FakeTitanicDataset(tmpdir)
                titanic_handle.download()
                mock_kaggle_api.authenticate.assert_called_once()
