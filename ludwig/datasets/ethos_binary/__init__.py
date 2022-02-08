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
from ludwig.datasets.mixins.process import IdentityProcessMixin
from ludwig.datasets.registry import register_dataset


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = EthosBinary(cache_dir=cache_dir)
    return dataset.load(split=split)


@register_dataset(name="ethos_binary")
class EthosBinary(UncompressedFileDownloadMixin, IdentityProcessMixin, CSVLoadMixin, BaseDataset):
    """The Ethos Hate Speech Dataset.

    Source Paper:
        ETHOS: an Online Hate Speech Detection Dataset
            Ioannis Mollas and Zoe Chrysopoulou and Stamatis Karlos and
            Grigorios Tsoumakas
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="ethos_binary", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        super().process_downloaded_dataset()
        # replace ; sperator to ,
        processed_df = pd.read_csv(os.path.join(self.processed_dataset_path, self.csv_filename), sep=";")
        # convert float labels (0.0, 1.0) to binary labels
        processed_df["isHate"] = processed_df["isHate"].astype(int)
        processed_df.to_csv(os.path.join(self.processed_dataset_path, self.csv_filename), index=False, sep=",")
