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
import pandas as pd
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import MultifileJoinProcessMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_api_key=None):
    dataset = AmesHousing(
        cache_dir=cache_dir,
        kaggle_username=kaggle_username,
        kaggle_api_key=kaggle_api_key
    )
    return dataset.load(split=split)


class AmesHousing(CSVLoadMixin, MultifileJoinProcessMixin, KaggleDownloadMixin, BaseDataset):
    """The Ames Housing dataset.

    Additional details:

    https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    """

    def __init__(self,
                 cache_dir=DEFAULT_CACHE_LOCATION,
                 kaggle_username=None,
                 kaggle_api_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_api_key = kaggle_api_key
        super().__init__(dataset_name='ames_housing', cache_dir=cache_dir)
