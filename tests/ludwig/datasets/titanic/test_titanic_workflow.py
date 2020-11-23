import tempfile
import pandas as pd
from unittest import mock
from ludwig.datasets.base_dataset import DEFAULT_CACHE_LOCATION
from ludwig.datasets.titanic import Titanic


class FakeTitanicDataset(Titanic):
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(cache_dir=cache_dir)


def test_download_titanic_dataset():
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
    titanic_train_filename = "titanic_train.csv"
    titanic_test_filename = "titanic_test.csv"
    with tempfile.TemporaryDirectory() as source_dir:
        titanic_train_df.to_csv(index=False)
        titanic_test_df.to_csv(index=False)

        config = {
            'version': 1.0,
            'split_filenames': {
                'train_file': titanic_train_filename,
                'test_file': titanic_test_filename
            },
            'csv_filename': 'fake_titanic.csv',
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch('ludwig.datasets.base_dataset.read_config',
                            return_value=config):
                dataset = FakeTitanicDataset(tmpdir)
                assert not dataset.is_downloaded()
                assert not dataset.is_processed()
                dataset.download()

                assert dataset.is_downloaded()
