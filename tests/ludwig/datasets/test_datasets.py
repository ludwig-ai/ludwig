import os
from unittest import mock

import pandas as pd
import pytest

import ludwig.datasets
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetState

SUPPORTED_UNCOMPRESSED_FILETYPES = ["json", "jsonl", "tsv", "csv"]


def test_load_csv_dataset(tmpdir):
    input_df = pd.DataFrame(
        {"name": ["Raphael", "Donatello"], "mask": ["red", "purple"], "weapon": ["sai", "bo staff"], "split": [0, 1]}
    )

    extracted_filename = "input.csv"
    compression_opts = dict(method="zip", archive_name=extracted_filename)

    archive_filename = os.path.join(tmpdir, "archive.zip")
    input_df.to_csv(archive_filename, index=False, compression=compression_opts)

    config = DatasetConfig(
        version=1.0,
        name="fake_csv_dataset",
        download_urls=["file://" + archive_filename],
    )

    ludwig.datasets._get_dataset_configs.cache_clear()
    with mock.patch("ludwig.datasets._load_dataset_config", return_value=config):
        dataset = ludwig.datasets.get_dataset("fake_csv_dataset", cache_dir=tmpdir)

        assert not dataset.state == DatasetState.DOWNLOADED
        assert not dataset.state == DatasetState.TRANSFORMED

        output_df = dataset.load()
        pd.testing.assert_frame_equal(input_df, output_df)

        assert dataset.state == DatasetState.TRANSFORMED
    ludwig.datasets._get_dataset_configs.cache_clear()


@pytest.mark.parametrize("f_type", SUPPORTED_UNCOMPRESSED_FILETYPES)
def test_multifile_join_dataset(tmpdir, f_type):
    if f_type != "jsonl":
        train_df = pd.DataFrame(
            {"name": ["Raphael", "Donatello"], "mask": ["red", "purple"], "weapon": ["sai", "bo staff"]}
        )

        test_df = pd.DataFrame({"name": ["Jack", "Bob"], "mask": ["green", "yellow"], "weapon": ["knife", "gun"]})

        val_df = pd.DataFrame({"name": ["Tom"], "mask": ["pink"], "weapon": ["stick"]})
    else:
        train_df = pd.DataFrame([{"name": "joe"}, {"mask": "green"}, {"weapon": "stick"}])
        test_df = pd.DataFrame([{"name": "janice"}, {"mask": "black"}, {"weapon": "gun"}])
        val_df = pd.DataFrame([{"name": "sara"}, {"mask": "pink"}, {"weapon": "gun"}])

    # filetypes = ['json', 'tsv', 'jsonl']
    train_filename = "train." + f_type
    test_filename = "test." + f_type
    val_filename = "val." + f_type
    train_filepath = os.path.join(tmpdir, train_filename)
    test_filepath = os.path.join(tmpdir, test_filename)
    val_filepath = os.path.join(tmpdir, val_filename)

    if f_type == "json":
        train_df.to_json(train_filepath)
        test_df.to_json(test_filepath)
        val_df.to_json(val_filepath)
    elif f_type == "jsonl":
        train_df.to_json(train_filepath, orient="records", lines=True)
        test_df.to_json(test_filepath, orient="records", lines=True)
        val_df.to_json(val_filepath, orient="records", lines=True)
    elif f_type == "tsv":
        train_df.to_csv(train_filepath, sep="\t")
        test_df.to_csv(test_filepath, sep="\t")
        val_df.to_csv(val_filepath, sep="\t")
    else:
        train_df.to_csv(train_filepath)
        test_df.to_csv(test_filepath)
        val_df.to_csv(val_filepath)

    config = DatasetConfig(
        version=1.0,
        name="fake_multifile_dataset",
        download_urls=["file://" + train_filepath, "file://" + test_filepath, "file://" + val_filepath],
        train_filenames=train_filename,
        validation_filenames=val_filename,
        test_filenames=test_filename,
    )

    ludwig.datasets._get_dataset_configs.cache_clear()
    with mock.patch("ludwig.datasets._load_dataset_config", return_value=config):
        dataset = ludwig.datasets.get_dataset("fake_multifile_dataset", cache_dir=tmpdir)

        assert not dataset.state == DatasetState.DOWNLOADED
        assert not dataset.state == DatasetState.TRANSFORMED

        output_df = dataset.load()
        assert output_df.shape[0] == train_df.shape[0] + test_df.shape[0] + val_df.shape[0]

        assert dataset.state == DatasetState.TRANSFORMED
    ludwig.datasets._get_dataset_configs.cache_clear()
