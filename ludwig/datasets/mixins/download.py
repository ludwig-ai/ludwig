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
import requests
import urllib.request
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


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
        """
                Download the raw dataset and contents of the gzip file
                onto the _raw directory.
                """
        os.makedirs(self.raw_temp_path, exist_ok=True)

        for file_download_url in self.download_urls:
            filename = file_download_url.split('/')[-1]
            response = requests.get(file_download_url, stream=True)
            size = len(filename)
            filename_root = filename[:size - 3]
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.raw.read())
            input_gzip_file = gzip.GzipFile(filename, 'rb')
            s = input_gzip_file.read()
            input_gzip_file.close()

            output = open(filename_root, 'wb')
            output.write(s)
            output.close()
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
            urllib.request.urlretrieve(url, os.path.join(self.raw_temp_path,filename))

        os.rename(self.raw_temp_path, self.raw_dataset_path)

    @property
    def download_url(self):
        return self.config["download_urls"]


