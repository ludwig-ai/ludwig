import os
import tempfile

from unittest import mock

import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset
from ludwig.datasets.mixins.download import ZipDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin


class FakeCSVDataset(ZipDownloadMixin, IdentityProcessMixin, CSVLoadMixin, BaseDataset):
    def __init__(self, cache_dir=None):
        super().__init__(dataset_name="fake", cache_dir=cache_dir)


def test_load_csv_dataset():
    input_df = pd.DataFrame({
        'name': ['Raphael', 'Donatello'],
        'mask': ['red', 'purple'],
        'weapon': ['sai', 'bo staff']
    })

    extracted_filename = 'input.csv'
    compression_opts = dict(
        method='zip',
        archive_name=extracted_filename
    )

    with tempfile.TemporaryDirectory() as source_dir:
        archive_filename = os.path.join(source_dir, 'archive.zip')
        input_df.to_csv(archive_filename, index=False, compression=compression_opts)

        config = dict(
            version=1.0,
            download_url='file://' + archive_filename,
            csv_filename=extracted_filename,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch('ludwig.datasets.base_dataset.read_config', return_value=config):
                dataset = FakeCSVDataset(tmpdir)

                assert not dataset.is_downloaded()
                assert not dataset.is_processed()

                output_df = dataset.load()
                pd.testing.assert_frame_equal(input_df, output_df)

                assert dataset.is_downloaded()
                assert dataset.is_processed()
