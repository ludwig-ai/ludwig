#! /usr/bin/env python
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

import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import ZipDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Naval(cache_dir=cache_dir)
    return dataset.load(split=split)


@register_dataset(name="naval")
class Naval(ZipDownloadMixin, CSVLoadMixin, BaseDataset):
    """Condition Based Maintenance of Naval Propulsion Plants Data Set.

    Additional Details:

    https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="naval", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        df = pd.read_csv(os.path.join(self.raw_dataset_path, "UCI CBM Dataset", "data.txt"), header=None, sep="   ")

        columns = [
            "lp",
            "v",
            "gtt",
            "gtn",
            "ggn",
            "ts",
            "tp",
            "t48",
            "t1",
            "t2",
            "p48",
            "p1",
            "p2",
            "pexh",
            "tic",
            "mf",
            "gtcdsc",
            "gttdsc",
        ]
        df.columns = columns

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_csv(os.path.join(self.processed_temp_path, self.csv_filename), index=False)
        rename(self.processed_temp_path, self.processed_dataset_path)
