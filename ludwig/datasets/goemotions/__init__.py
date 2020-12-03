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
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import UncompressedFileDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import *


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = GoEmotions(cache_dir=cache_dir)
    return dataset.load(split=split)


class GoEmotions(UncompressedFileDownloadMixin, MultifileJoinProcessMixin,
                 CSVLoadMixin, BaseDataset):
    """The GoEmotions dataset.

    This pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming training data into a destination
    dataframe that can fit into Ludwig's training API.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="goemotions", cache_dir=cache_dir)

    def read_file(self, filetype, filename):
        file_df = pd.read_table(os.path.join(self.raw_dataset_path, filename),
                                header=None)
        return file_df

    def process_downloaded_dataset(self):
        super(GoEmotions, self).process_downloaded_dataset()
        # format emotion ids to be a set of emotion ids vs. string
        processed_df = pd.read_csv(os.path.join(self.processed_dataset_path,
                                                self.csv_filename))
        processed_df.columns = ['text', 'emotion_ids', 'comment_id', 'split']
        processed_df['emotion_ids'] = processed_df['emotion_ids'].apply(
            lambda e_id: " ".join(e_id.split(","))
        )
        processed_df.to_csv(
            os.path.join(self.processed_dataset_path, self.csv_filename),
            index=False
        )
