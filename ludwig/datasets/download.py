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
from io import BytesIO
from urllib.request import urlopen
from pathlib import Path
from zipfile import ZipFile


class ZipDownloadMixin:
    """A mixin to simulate downloading the zip file containing the training data
    and extracting the contents"""

    """Download the raw dataset and extract the contents
    of the zip file and store that in the cache location."""
    def download_raw_dataset(self):
        _raw_file_name = Path.home().joinpath('.ludwig_cache').joinpath(self._dataset_name + "_"
                                                                             + str(self._dataset_version)). \
            joinpath('raw.csv')
        _download_dir = os.path.join(self._cache_location, f'{self._dataset_name}_{self._dataset_version}')
        with urlopen(self._download_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(self._download_dir)
        # we downloaded the file, now check that this file exists
        downloaded_file = _download_dir.joinpath(self._dataset_file_name)

        # rename the file to raw.csv
        if os.path.isfile(downloaded_file):
            os.rename(downloaded_file, _raw_file_name)

