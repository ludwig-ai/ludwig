#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import gzip
import os
import shutil
import tarfile
import urllib.request
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Provides progress bar for `urlretrieve`.

    Taken from: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class ZipDownloadMixin:
    """Downloads the zip file containing the training data and extracts the contents."""

    config: dict
    raw_dataset_path: str
    raw_temp_path: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.
        """
        os.makedirs(self.raw_temp_path, exist_ok=True)
        for url in self.download_urls:
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(self.raw_temp_path)
        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @property
    def download_urls(self):
        return self.config["download_urls"]

class TarDownloadMixin:
    """Downloads the compressed tar file containing the training data and extracts the contents."""

    config: dict
    raw_dataset_path: str
    raw_temp_path: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the tar file and
        store that in the cache location.
        """
        os.makedirs(self.raw_temp_path, exist_ok=True)
        for url in self.download_urls:
            filename = url.split('/')[-1]
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024,
                          miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(
                    url,
                    os.path.join(self.raw_temp_path, filename),
                    t.update_to
                )

            download_folder_name = url.split('/')[-1].split('.')[0]
            file_path = os.path.join(self.raw_temp_path, filename)
            with tarfile.open(file_path) as tar_file:
                tar_file.extractall(path=self.raw_temp_path)

            for f in os.scandir(os.path.join(self.raw_temp_path, 
                                             download_folder_name)):
                shutil.copyfile(f, os.path.join(self.raw_temp_path, f.name))

        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @property
    def download_urls(self):
        return self.config['download_urls']

class GZipDownloadMixin:
    """Downloads the gzip archive file containing the training data and extracts the contents."""

    config: dict
    raw_dataset_path: str
    raw_temp_path: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.
        """
        os.makedirs(self.raw_temp_path, exist_ok=True)
        for file_download_url in self.download_urls:
            filename = file_download_url.split('/')[-1]
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024,
                          miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(
                    file_download_url,
                    os.path.join(self.raw_temp_path, filename),
                    t.update_to
                )
            gzip_content_file = '.'.join(filename.split('.')[:-1])
            with gzip.open(
                    os.path.join(self.raw_temp_path, filename)) as gzfile:
                with open(os.path.join(self.raw_temp_path, gzip_content_file),
                          'wb') as output:
                    shutil.copyfileobj(gzfile, output)
        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @property
    def download_urls(self):
        return self.config["download_urls"]


class UncompressedFileDownloadMixin:
    """Downloads the json file containing the training data and extracts the contents."""

    config: dict
    raw_dataset_path: str
    raw_temp_path: str

    def download_raw_dataset(self):
        """
        Download the raw dataset files and store in the cache location.
        """
        os.makedirs(self.raw_temp_path, exist_ok=True)
        for url in self.download_url:
            filename = url.split('/')[-1]
            urllib.request.urlretrieve(url, os.path.join(self.raw_temp_path,
                                                         filename))

        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @property
    def download_url(self):
        return self.config["download_urls"]
