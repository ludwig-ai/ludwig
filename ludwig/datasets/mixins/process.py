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

import pandas as pd


class IdentityProcessMixin:
    """A mixin that performs a no-op for already processed raw datasets."""

    raw_dataset_path: str
    processed_dataset_path: str

    def process_downloaded_dataset(self):
        os.rename(self.raw_dataset_path, self.processed_dataset_path)


class MultifileJoinProcessMixin:
    """A mixin that joins raw files to build final dataset"""

    config: dict
    raw_dataset_path: str
    processed_dataset_path: str

    def read_file(self, filetype, filename, header):
        if filetype == 'json':
            file_df = pd.read_json(
                os.path.join(self.raw_dataset_path, filename))
        elif filetype == 'jsonl':
            file_df = pd.read_json(
                os.path.join(self.raw_dataset_path, filename), lines=True)
        elif filetype == 'tsv':
            file_df = pd.read_table(
                os.path.join(self.raw_dataset_path, filename))
        elif filetype == 'csv':
            file_df = pd.read_csv(
                os.path.join(self.raw_dataset_path, filename), header=header)
        else:
            raise ValueError(f'Unsupported file type: {filetype}')
        return file_df

    def process_downloaded_dataset(self, header=0):
        """Processes dataset

        :param header: indicates whether raw  data files contain headers
        """
        downloaded_files = self.download_filenames
        filetype = self.download_file_type
        all_files = []
        for split_name, filename in downloaded_files.items():
            file_df = self.read_file(filetype, filename, header)
            if split_name == 'train_file':
                file_df['split'] = 0
            elif split_name == 'val_file':
                file_df['split'] = 1
            elif split_name == 'test_file':
                file_df['split'] = 2
            else:
                raise ValueError(f'Unrecognized split name: {split_name}')
            all_files.append(file_df)

        concat_df = pd.concat(all_files, ignore_index=True)
        os.makedirs(self.processed_dataset_path, exist_ok=True)
        concat_df.to_csv(
            os.path.join(self.processed_dataset_path, self.csv_filename),
            index=False)

    @property
    def download_filenames(self):
        return self.config['split_filenames']

    @property
    def download_file_type(self):
        return self.config['download_file_type']

    @property
    def csv_filename(self):
        return self.config['csv_filename']
