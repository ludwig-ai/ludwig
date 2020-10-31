import os
import pytest
import tempfile
import pandas as pd

from unittest import mock


from ludwig.datasets.base_dataset import BaseDataset
from ludwig.datasets.mixins.download import GZipDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin


class FakeMnistDataset(GZipDownloadMixin, IdentityProcessMixin, CSVLoadMixin, BaseDataset):
    def __init__(self, cache_dir=None):
        super().__init__(dataset_name="mnist", cache_dir=cache_dir)


def test_load_mnist_dataset():
    input_df = pd.DataFrame({
        'image_path': ['training/0/1234.png', 'training/0/2345.png'],
        'label': [0, 0]
    })

    extracted_filenames = ['train-images-idx3-ubyte',
                           'train-labels-idx1-ubyte',
                           't10k-images-idx3-ubyte',
                           't10k-labels-idx1-ubyte']
    compression_opts = dict(
        method='gzip'
    )

    with tempfile.TemporaryDirectory() as source_dir:
        archive_filename = os.path.join(source_dir, 'archive.zip')
        input_df.to_csv(archive_filename, index=False, compression=compression_opts)
        download_urls= [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

        config = dict(
            version=1.0,
            download_urls=download_urls,
            csv_filename=extracted_filenames,
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
