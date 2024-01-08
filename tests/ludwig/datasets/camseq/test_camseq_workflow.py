import os
import zipfile
from shutil import copy
from unittest import mock

import ludwig.datasets
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetState


def test_download_camseq_dataset(tmpdir):
    image_filename = os.path.join(tmpdir, "0123.png")
    image_handle = open(image_filename, "w+b")
    image_handle.write(b"This binary string will be written as image data")
    image_handle.close()

    mask_filename = os.path.join(tmpdir, "0123_L.png")
    mask_handle = open(mask_filename, "w+b")
    mask_handle.write(b"This binary string will be written as mask data")
    mask_handle.close()

    archive_filename = os.path.join(tmpdir, "camseq-semantic-segmentation.zip")
    with zipfile.ZipFile(archive_filename, "w") as arc_zip:
        arc_zip.write(image_filename, "0123.png")
        arc_zip.write(mask_filename, "0123_L.png")

    config = DatasetConfig(
        version=1.0,
        name="camseq",
        kaggle_dataset_id="carlolepelaars/camseq-semantic-segmentation",
        archive_filenames="camseq-semantic-segmentation.zip",
        # Normally we would verify the zip file, but in this test the zip file is created every time and contains the
        # creation dates of the image files so its digest will be different every time the test is run.
        sha256={
            "0123.png": "c1596140553bd796fdc77369c433de07ef41779a0defa412a9c204ed71f4697c",
            "0123_L.png": "f75ca47a9b6d3314008f3d9de853b8af1650503ac16ab406a7908832cdf57168",
        },
        loader="camseq.CamseqLoader",
        preserve_paths=["images", "masks"],
    )

    def download_files(dataset_id, path):
        assert dataset_id == "carlolepelaars/camseq-semantic-segmentation"
        copy(archive_filename, path)

    ludwig.datasets._get_dataset_configs.cache_clear()
    with mock.patch("ludwig.datasets._load_dataset_config", return_value=config):
        with mock.patch("ludwig.datasets.kaggle.create_kaggle_client") as mock_kaggle_cls:
            mock_kaggle_api = mock.MagicMock()
            mock_kaggle_api.dataset_download_files = download_files
            mock_kaggle_cls.return_value = mock_kaggle_api

            dataset = ludwig.datasets.get_dataset("camseq", cache_dir=tmpdir)
            assert not dataset.state == DatasetState.DOWNLOADED

            dataset.download()
            assert dataset.state == DatasetState.DOWNLOADED
            mock_kaggle_api.authenticate.assert_called_once()

            assert not dataset.state == DatasetState.TRANSFORMED
            dataset.extract()
            # Normally we would verify before extracting, but in this test the zip file is created on each run and
            # changes between test runs. Instead we verify the extracted image files.
            dataset.verify()

            df = dataset.load(split=False)
            assert len(df) == 1
    ludwig.datasets._get_dataset_configs.cache_clear()
