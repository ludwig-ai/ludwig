import gzip
import os
import shutil
from unittest import mock

import ludwig.datasets
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetState


def test_download_av_mnist_dataset(tmpdir):
    train_image_archive_filename = os.path.join(tmpdir, "train-images-idx3-ubyte")
    train_image_handle = open(train_image_archive_filename, "w+b")
    train_image_handle.write(b"This binary string will be written as training mage data")
    train_image_handle.close()
    with open(train_image_archive_filename, "rb") as f_in:
        with gzip.open(train_image_archive_filename + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    train_labels_archive_filename = os.path.join(tmpdir, "train-labels-idx1-ubyte")
    train_labels_handle = open(train_labels_archive_filename, "w")
    train_labels_handle.write("0")
    train_labels_handle.close()
    with open(train_labels_archive_filename, "rb") as f_in:
        with gzip.open(train_labels_archive_filename + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    test_image_archive_filename = os.path.join(tmpdir, "t10k-images-idx3-ubyte")
    test_image_handle = open(test_image_archive_filename, "w+b")
    test_image_handle.write(b"This binary string will be written as test mage data")
    test_image_handle.close()
    with open(test_image_archive_filename, "rb") as f_in:
        with gzip.open(test_image_archive_filename + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    test_labels_archive_filename = os.path.join(tmpdir, "t10k-labels-idx1-ubyte")
    test_labels_handle = open(test_labels_archive_filename, "w")
    test_labels_handle.write("0")
    test_labels_handle.close()
    with open(test_labels_archive_filename, "rb") as f_in:
        with gzip.open(test_labels_archive_filename + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    download_urls = [
        "file://" + train_image_archive_filename + ".gz",
        "file://" + train_labels_archive_filename + ".gz",
        "file://" + test_image_archive_filename + ".gz",
        "file://" + test_labels_archive_filename + ".gz",
    ]

    config = DatasetConfig(
        version=1.0,
        name="av_mnist",
        download_urls=download_urls,
    )

    ludwig.datasets._get_dataset_configs.cache_clear()
    with mock.patch("ludwig.datasets._load_dataset_config", return_value=config):
        dataset = ludwig.datasets.get_dataset("av_mnist", cache_dir=tmpdir)
        assert not dataset.state == DatasetState.DOWNLOADED
        assert not dataset.state == DatasetState.TRANSFORMED
        dataset.download()

        assert dataset.state == DatasetState.DOWNLOADED
    ludwig.datasets._get_dataset_configs.cache_clear()
