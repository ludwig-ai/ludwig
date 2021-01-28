import tempfile
import pandas as pd
import os
import zipfile
from shutil import copy
from unittest import mock

from ludwig.datasets.titanic import Titanic


class FakeTitanicDataset(Titanic):
    def __init__(self, cache_dir):
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
            z.write(train_fname, 'train.csv')
            z.write(test_fname, 'test.csv')

        config = {
            'version': 1.0,
            'competition': 'titanic',
            'archive_filename': 'titanic.zip',
            'split_filenames': {
                'train_file': 'train.csv',
                'test_file': 'test.csv',
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

                dataset = FakeTitanicDataset(tmpdir)
                assert not dataset.is_downloaded()

                dataset.download()
                assert dataset.is_downloaded()
                mock_kaggle_api.authenticate.assert_called_once()

                assert not dataset.is_processed()
                dataset.process()
                assert dataset.is_processed()

                output_train_df, output_test_df, output_val_df = dataset.load(split=True)
                assert len(output_train_df) == len(titanic_train_df)
                assert len(output_test_df) == len(titanic_test_df)
                assert len(output_val_df) == 0
