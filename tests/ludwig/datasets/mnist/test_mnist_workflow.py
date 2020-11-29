import gzip
import os
import shutil
import tempfile
from unittest import mock

from ludwig.datasets.mnist import Mnist


class FakeMnistDataset(Mnist):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)


def test_download_mnist_dataset():
    with tempfile.TemporaryDirectory() as source_dir:
        train_image_archive_filename = os.path.join(source_dir,
                                                    'train-images-idx3-ubyte')
        train_image_handle = open(train_image_archive_filename, 'w+b')
        train_image_handle.write(
            b"This binary string will be written as training mage data")
        train_image_handle.close()
        with open(train_image_archive_filename, 'rb') as f_in:
            with gzip.open(train_image_archive_filename + ".gz",
                           'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        train_labels_archive_filename = os.path.join(source_dir,
                                                     'train-labels-idx1-ubyte')
        train_labels_handle = open(train_labels_archive_filename, 'w')
        train_labels_handle.write("0")
        train_labels_handle.close()
        with open(train_labels_archive_filename, 'rb') as f_in:
            with gzip.open(train_labels_archive_filename + ".gz",
                           'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        test_image_archive_filename = os.path.join(source_dir,
                                                   't10k-images-idx3-ubyte')
        test_image_handle = open(test_image_archive_filename, 'w+b')
        test_image_handle.write(
            b"This binary string will be written as test mage data")
        test_image_handle.close()
        with open(test_image_archive_filename, 'rb') as f_in:
            with gzip.open(test_image_archive_filename + ".gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        test_labels_archive_filename = os.path.join(source_dir,
                                                    't10k-labels-idx1-ubyte')
        test_labels_handle = open(test_labels_archive_filename, 'w')
        test_labels_handle.write("0")
        test_labels_handle.close()
        with open(test_labels_archive_filename, 'rb') as f_in:
            with gzip.open(test_labels_archive_filename + ".gz",
                           'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        extracted_filenames = ['train-images-idx3-ubyte',
                               'train-labels-idx1-ubyte',
                               't10k-images-idx3-ubyte',
                               't10k-labels-idx1-ubyte']

        download_urls = ['file://' + train_image_archive_filename + ".gz",
                         'file://' + train_labels_archive_filename + ".gz",
                         'file://' + test_image_archive_filename + ".gz",
                         'file://' + test_labels_archive_filename + ".gz"]

        config = dict(
            version=1.0,
            download_urls=download_urls,
            csv_filename=extracted_filenames,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch('ludwig.datasets.base_dataset.read_config',
                            return_value=config):
                dataset = FakeMnistDataset(tmpdir)
                assert not dataset.is_downloaded()
                assert not dataset.is_processed()
                dataset.download()

                assert dataset.is_downloaded()
