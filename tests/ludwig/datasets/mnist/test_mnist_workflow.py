import os
import pytest
import tempfile
import pandas as pd
from unittest import mock
from ludwig.datasets.mnist import Mnist
from ludwig.datasets.mixins.download import GZipDownloadMixin


class FakeMnistDataset(GZipDownloadMixin, Mnist):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)


def test_load_mnist_dataset():
    input_df = pd.DataFrame({
        'image_path': ['training/0/1234.png', 'training/0/2345.png'],
        'label': [0, 0]
    })

    download_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    output_filename = 'mnist_dataset_testing.csv'
    compression_opts = dict(
        method='gzip'
    )

    with tempfile.TemporaryDirectory() as source_dir:
        archive_filename = os.path.join(source_dir, 'train-images-idx3-ubyte.gz')
        input_df.to_csv(archive_filename, index=False, compression=compression_opts)
        config = dict(
            version=1.0,
            download_urls=download_urls,
            csv_filename=output_filename,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch('ludwig.datasets.base_dataset.read_config', return_value=config):
                dataset = FakeMnistDataset(tmpdir)

                assert not dataset.is_downloaded()
                assert not dataset.is_processed()

                output_df = dataset.load()
                pd.testing.assert_frame_equal(input_df, output_df)

                assert dataset.is_downloaded()
                assert dataset.is_processed()
