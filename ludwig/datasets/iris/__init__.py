#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    if split:
        raise ValueError("Iris dataset does not contain a split column")
    dataset = Iris(cache_dir=cache_dir)
    return dataset.load(split=split)


@register_dataset(name="iris")
class Iris(UncompressedFileDownloadMixin, CSVLoadMixin, BaseDataset):
    """The Iris dataset.

    Additional Details:

    https://archive.ics.uci.edu/ml/datasets/Iris
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="iris", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        raw_df = pd.read_csv(os.path.join(self.raw_dataset_path, self.data_filename))
        columns = [
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
            "class",
        ]
        raw_df.columns = columns

        makedirs(self.processed_dataset_path, exist_ok=True)
        raw_df.to_csv(os.path.join(self.processed_dataset_path, self.csv_filename), index=False)

    @property
    def data_filename(self):
        return self.config["data_filename"]
