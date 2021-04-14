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
from scipy.io import loadmat
import pandas as pd
from typing import Tuple, Union

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import MultifileJoinProcessMixin
from ludwig.constants import SPLIT


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True, kaggle_username=None, kaggle_key=None):
    dataset = RossmannStoreSales(
        cache_dir=cache_dir,
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
    )
    return dataset.load(split=split)
    
class RossmannStoreSales(KaggleDownloadMixin, MultifileJoinProcessMixin,
            CSVLoadMixin, BaseDataset):
    """
        The Rossmann Store Sales dataset

        Additional Details:

        https://www.kaggle.com/c/rossmann-store-sales
    """
    def __init__(self,
                 cache_dir=DEFAULT_CACHE_LOCATION,
                 kaggle_username=None,
                 kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        super().__init__(dataset_name='rossmann_store_sales', cache_dir=cache_dir)

    def load_processed_dataset(self, split) -> Union[pd.DataFrame,
                                                     Tuple[pd.DataFrame,
                                                           pd.DataFrame]]:
        
        train_df, test_df = super().load_processed_dataset(split=split)

        training_set = train_df[train_df[SPLIT] == 0]
        training_set = training_set.dropna(axis=1)

        test_set = test_df[test_df[SPLIT] == 2]
        test_set = test_set.dropna(axis=1)
        return training_set, test_set        