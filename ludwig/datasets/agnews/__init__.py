#! /usr/bin/env python
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

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import UncompressedFileDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import MultifileJoinProcessMixin
from ludwig.datasets.registry import register_dataset


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True):
    dataset = AGNews(cache_dir=cache_dir)
    return dataset.load(split=split)


@register_dataset(name="agnews")
class AGNews(UncompressedFileDownloadMixin, MultifileJoinProcessMixin, CSVLoadMixin, BaseDataset):
    """The AGNews dataset."""

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="agnews", cache_dir=cache_dir)

    def read_file(self, filetype, filename, header=0):
        file_df = pd.read_csv(os.path.join(self.raw_dataset_path, filename))
        # class_index : number between 1-4 where
        # 1-World, 2-Sports, 3-Business, 4-Science/Tech
        file_df.columns = ["class_index", "title", "description"]
        return file_df

    def process_downloaded_dataset(self):
        super().process_downloaded_dataset(header=None)
        processed_df = pd.read_csv(os.path.join(self.processed_dataset_path, self.csv_filename))
        # Maps class_index to class name.
        class_names = ["", "world", "sports", "business", "sci_tech"]
        # Adds new column 'class' by mapping class indexes to strings.
        processed_df["class"] = processed_df.class_index.apply(lambda i: class_names[i])
        # Agnews has no validation split, only train and test (0, 2). For convenience, we'll designate the first 5% of
        # each class from the training set as the validation set.
        val_set_n = int((len(processed_df) * 0.05) // len(class_names))  # rows from each class in validation set.
        for ci in range(1, 5):
            # For each class, reassign the first val_set_n rows of the training set to validation set.
            train_rows = processed_df[(processed_df.split == 0) & (processed_df.class_index == ci)].index
            processed_df.loc[train_rows[:val_set_n], "split"] = 1
        processed_df.to_csv(os.path.join(self.processed_dataset_path, self.csv_filename), index=False)
