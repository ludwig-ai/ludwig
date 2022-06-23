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
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin
from ludwig.datasets.registry import register_dataset


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_api_key=None):
    dataset = Temperature(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_api_key=kaggle_api_key)
    return dataset.load(split=split)


@register_dataset(name="temperature")
class Temperature(CSVLoadMixin, IdentityProcessMixin, KaggleDownloadMixin, BaseDataset):
    """Hourly temperature dataset from Kaggle."""

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_api_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_api_key = kaggle_api_key
        self.is_kaggle_competition = False
        super().__init__(dataset_name="temperature", cache_dir=cache_dir)
