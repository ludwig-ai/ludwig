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
from zipfile import ZipFile


"""A mixin to simulate downloading the zip file containing the training data
and extracting the contents"""


class ZipDownloadMixin:

    """Download the raw dataset and extract the contents
    of the zip file and store that in the cache location.
    args:
        None
    ret:
        None"""
    def downloaded_raw_dataset(self):
        with urlopen(self._download_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(self._download_dir)
        # we downloaded the file, now check that this file exists
        downloaded_file = self._download_dir.joinpath(self._dataset_file_name)

        # rename the file to raw.csv
        if os.path.isfile(downloaded_file):
            os.rename(downloaded_file, self._raw_file_name)

