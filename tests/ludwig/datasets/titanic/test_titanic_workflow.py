import os
import zipfile
from shutil import copy
from unittest import mock

import pandas as pd

import ludwig.datasets
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetState


def test_download_titanic_dataset(tmpdir):
    titanic_train_df = pd.DataFrame(
        {
            "passenger_id": [1216, 699, 234],
            "pclass": [3, 3, 4],
            "name": ["sai bo", "bo staff", "tae kwan nic"],
            "sex": ["female", "male", "male"],
            "age": [38, 28, 18],
            "sibsp": [0, 1, 0],
            "parch": [1, 1, 2],
            "ticket": [335432, 315089, 322472],
            "fare": [7.7333, 8.6625, 9.8765],
            "cabin": [1, 2, 4],
            "embarked": ["C", "Q", "S"],
            "boat": [0, 0, 0],
            "body": [0, 1, 0],
            "home.dest": ["Croatia", "Italy", "Sweden"],
            "survived": [0, 1, 0],
        }
    )

    titanic_test_df = pd.DataFrame(
        {
            "passenger_id": [1216, 699, 234],
            "pclass": [3, 3, 4],
            "name": ["mo bo", "bo bo bo", "Rafael Nadal"],
            "sex": ["female", "male", "male"],
            "age": [28, 18, 30],
            "sibsp": [0, 1, 0],
            "parch": [1, 1, 2],
            "ticket": [335412, 215089, 922472],
            "fare": [17.7333, 18.6625, 19.8765],
            "cabin": [2, 2, 1],
            "embarked": ["Q", "Q", "C"],
            "boat": [0, 0, 0],
            "body": [0, 1, 0],
            "home.dest": ["Sweden", "Slovenia", "Italy"],
            "survived": [0, 1, 0],
        }
    )

    train_fname = os.path.join(tmpdir, "train.csv")
    titanic_train_df.to_csv(train_fname, index=False)

    test_fname = os.path.join(tmpdir, "test.csv")
    titanic_test_df.to_csv(test_fname, index=False)

    archive_filename = os.path.join(tmpdir, "titanic.zip")
    with zipfile.ZipFile(archive_filename, "w") as z:
        z.write(train_fname, "train.csv")
        z.write(test_fname, "test.csv")

    config = DatasetConfig(
        version=1.0,
        name="titanic",
        kaggle_competition="titanic",
        archive_filenames="titanic.zip",
        # Normally we would verify the zip file, but in this test the zip file is created every time and contains the
        # creation dates of the csv files so its digest will be different every time the test is run.
        sha256={
            "test.csv": "348c49a95fe099fcc3b9142c82fb6becb87edc0f4d2c69c485e0dce4af8625e0",
            "train.csv": "483556c465414fd78deb02b25f39a0de844b0728c1ef0505df0e5b3e40fec995",
        },
        train_filenames="train.csv",
        test_filenames="test.csv",
    )

    def download_files(competition_name, path):
        assert competition_name == "titanic"
        copy(archive_filename, path)

    ludwig.datasets._get_dataset_configs.cache_clear()
    with mock.patch("ludwig.datasets._load_dataset_config", return_value=config):
        with mock.patch("ludwig.datasets.kaggle.create_kaggle_client") as mock_kaggle_cls:
            mock_kaggle_api = mock.MagicMock()
            mock_kaggle_api.competition_download_files = download_files
            mock_kaggle_cls.return_value = mock_kaggle_api

            dataset = ludwig.datasets.get_dataset("titanic", cache_dir=tmpdir)
            assert not dataset.state == DatasetState.DOWNLOADED

            dataset.download()
            assert dataset.state == DatasetState.DOWNLOADED
            mock_kaggle_api.authenticate.assert_called_once()

            assert not dataset.state == DatasetState.TRANSFORMED
            dataset.extract()
            # Normally we would verify before extracting, but in this test the zip file is created on each run and
            # changes between test runs. Instead we verify the extracted .csv files.
            dataset.verify()
            dataset.transform()
            assert dataset.state == DatasetState.TRANSFORMED

            output_train_df, output_test_df, output_val_df = dataset.load(split=True)
            assert len(output_train_df) == len(titanic_train_df)
            assert len(output_test_df) == len(titanic_test_df)
            assert len(output_val_df) == 0
    ludwig.datasets._get_dataset_configs.cache_clear()
