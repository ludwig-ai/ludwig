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
import os
import tempfile

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


class ZipDownloadMixin:
    """Downloads the zip file containing the training data and extracts the contents."""

    config: dict
    raw_dataset_path: str
    download_dir: str

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with urlopen(self.download_url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(tmpdir)

            local_filename = os.path.join(tmpdir, self.extracted_filename)
            if not os.path.exists(local_filename):
                raise RuntimeError(f'Expected extracted file {local_filename} does not exist')

            os.makedirs(self.download_dir, exist_ok=True)
            os.rename(local_filename, self.raw_dataset_path)

    def is_downloaded(self) -> bool:
        return os.path.exists(self.raw_dataset_path)

    @property
    def download_url(self):
        return self.config["download_url"]

    @property
    def extracted_filename(self):
        return self.config["extracted_file_name"]
