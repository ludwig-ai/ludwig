import io
import os
import uuid
from unittest import mock

import pandas as pd
import pytest

import ludwig.datasets
from ludwig.api import LudwigModel
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


def test_get_datasets_info():
    dataset_output_features = ludwig.datasets.get_datasets_output_features()
    assert len(dataset_output_features) > 1
    assert isinstance(dataset_output_features, dict)
    assert dataset_output_features["titanic"].get("name", None)
    assert dataset_output_features["titanic"].get("output_features", None)
    assert isinstance(dataset_output_features["titanic"]["output_features"], list)
    assert dataset_output_features["titanic"]["output_features"][0].get("name", None)
    assert dataset_output_features["titanic"]["output_features"][0].get("type", None)

    dataset_output_features = ludwig.datasets.get_datasets_output_features(dataset="titanic")
    assert len(dataset_output_features["output_features"]) == 1
    assert dataset_output_features["name"] == "titanic"


def test_get_dataset_buffer():
    buffer = ludwig.datasets.get_buffer("iris")

    assert isinstance(buffer, io.BytesIO)


def test_preprocess_dataset_uri(tmpdir):
    input_df = pd.DataFrame(
        {
            "input": ["a", "b", "a", "b", "a", "b", "c", "c", "a", "b"],
            "output": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "split": [0, 0, 0, 0, 0, 0, 0, 1, 2, 2],
        }
    )

    extracted_filename = "input.csv"
    compression_opts = dict(method="zip", archive_name=extracted_filename)

    archive_filename = os.path.join(tmpdir, "archive.zip")
    input_df.to_csv(archive_filename, index=False, compression=compression_opts)

    dataset_name = f"fake_csv_dataset_{uuid.uuid4().hex}"
    config = DatasetConfig(
        version=1.0,
        name=dataset_name,
        download_urls=["file://" + archive_filename],
    )

    model_config = {
        "input_features": [{"name": "input", "type": "category"}],
        "output_features": [{"name": "output", "type": "number"}],
        "preprocessing": {"split": {"type": "fixed"}},
    }

    ludwig.datasets._get_dataset_configs.cache_clear()
    with mock.patch("ludwig.datasets._load_dataset_config", return_value=config):
        with mock.patch("ludwig.datasets.loaders.dataset_loader.get_default_cache_location", return_value=str(tmpdir)):
            model = LudwigModel(model_config, backend="local")

            proc_result = model.preprocess(dataset=f"ludwig://{dataset_name}")
            train_df1 = proc_result.training_set.to_df()
            val_df1 = proc_result.validation_set.to_df()
            test_df1 = proc_result.test_set.to_df()

            assert len(train_df1) == 7
            assert len(val_df1) == 1
            assert len(test_df1) == 2

            proc_result_split = model.preprocess(
                training_set=f"ludwig://{dataset_name}",
                validation_set=f"ludwig://{dataset_name}",
                test_set=f"ludwig://{dataset_name}",
            )
            train_df2 = proc_result_split.training_set.to_df()
            val_df2 = proc_result_split.validation_set.to_df()
            test_df2 = proc_result_split.test_set.to_df()

            assert len(train_df2) == 7
            assert len(val_df2) == 1
            assert len(test_df2) == 2

            assert train_df1.equals(train_df2)
            assert val_df1.equals(val_df2)
            assert test_df1.equals(test_df2)

    ludwig.datasets._get_dataset_configs.cache_clear()
