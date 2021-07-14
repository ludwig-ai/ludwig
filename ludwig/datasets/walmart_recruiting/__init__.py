#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Uber Technologies, Inc.
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
from zipfile import ZipFile

import pandas as pd
from ludwig.datasets.base_dataset import DEFAULT_CACHE_LOCATION, BaseDataset
from ludwig.datasets.mixins.kaggle import (KaggleDownloadMixin,
                                           create_kaggle_client)
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_key=None):
    dataset = WalmartRecruiting(
        cache_dir=cache_dir,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key
    )
    return dataset.load(split=split)


class WalmartRecruiting(CSVLoadMixin, IdentityProcessMixin, KaggleDownloadMixin, BaseDataset):
    """The Walmart Recruiting: Trip Type Classification
    https://www.kaggle.com/c/walmart-recruiting-trip-type-classification
    """

    def __init__(self,
                 cache_dir=DEFAULT_CACHE_LOCATION,
                 kaggle_username=None,
                 kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = True
        super().__init__(dataset_name='walmart_recruiting', cache_dir=cache_dir)

    def download_raw_dataset(self):
        """
        Download the raw dataset and extract the contents of the zip file and
        store that in the cache location.  If the user has not specified creds in the
        kaggle.json file we lookup the passed in username and the api key and
        perform authentication.
        """
        with self.update_env(KAGGLE_USERNAME=self.kaggle_username, KAGGLE_KEY=self.kaggle_key):
            # Call authenticate explicitly to pick up new credentials if necessary
            api = create_kaggle_client()
            api.authenticate()
        os.makedirs(self.raw_temp_path, exist_ok=True)

        if self.is_kaggle_competition:
            download_func = api.competition_download_files
        else:
            download_func = api.dataset_download_files
        # Download all files for a competition/dataset
        download_func(self.competition_name, path=self.raw_temp_path)

        archive_zip = os.path.join(self.raw_temp_path, self.archive_filename)
        print(archive_zip)
        # test.csv.zip is an encrypted zip file that requires a password
        # Avoid unzipping that file
        with ZipFile(archive_zip, 'r') as z:
            file_names = z.namelist()
            for fname in file_names:
                if fname != "test.csv.zip":
                    z.extract(fname, self.raw_temp_path)
        os.rename(self.raw_temp_path, self.raw_dataset_path)
