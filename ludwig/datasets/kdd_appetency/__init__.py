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
from typing import Tuple

import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import ZipandUncompressedFileDownloadMixin
from ludwig.datasets.mixins.process import MultifileJoinProcessMixin
from ludwig.constants import SPLIT


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True):
    dataset = KDDAppetency(cache_dir=cache_dir)
    return dataset.load(split=split)

class KDDAppetency(ZipandUncompressedFileDownloadMixin, MultifileJoinProcessMixin,
                   BaseDataset):
    """
    The KDD Cup 2009 Appetency dataset

    Additional Details:

    https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, label=None):
        super().__init__(dataset_name="kdd_appetency", cache_dir=cache_dir)

    def read_file(self, filetype, filename, header=0):
        if filetype == 'data':
            file_df = pd.read_csv(
                os.path.join(self.raw_dataset_path, filename), header=header, sep='\t')
        elif filetype == 'labels':
            file_df = -pd.read_csv(
                os.path.join(self.raw_dataset_path, filename), header=None)[0]
        elif filetype == 'txt':
            file_df = pd.read_csv(
                os.path.join(self.raw_dataset_path, filename), header=None)
        return file_df

    def process_downloaded_dataset(self, header=0):
        super().process_downloaded_dataset()
        unprocessed_df = pd.read_csv(os.path.join(self.processed_dataset_path,
                                                self.csv_filename))
        
        train_df = unprocessed_df[unprocessed_df[SPLIT] == 0]
        self.test_df = unprocessed_df[unprocessed_df[SPLIT] == 2]
        
        categorical_features = { 190, 191, 192, 193, 194, 195, 196, 197, 198,
                                199, 200, 201, 202, 203, 204, 205, 206, 207,
                                209, 210, 211, 212, 213, 214, 215, 216, 217,
                                218, 219, 220, 221, 222, 223, 224, 225, 226,
                                227, 228 }

        train_df = process_categorical_features(train_df, categorical_features)
        train_df = process_numerical_features(train_df, categorical_features)
        
        target = self.read_file('labels', os.path.join(self.raw_dataset_path,
                                                       'orange_small_train_appetency.labels'))
        
        train_idx = self.read_file('txt', os.path.join(self.raw_dataset_path,
                                                       'stratified_train_idx_appetency.txt'))
        
        val_idx = self.read_file('txt', os.path.join(self.raw_dataset_path,
                                                       'stratified_test_idx_appetency.txt'))
        
        processed_train_df = train_df.iloc[train_idx[0]].copy()
        processed_train_df['target'] = target.iloc[train_idx[0]]
        
        processed_val_df = train_df.iloc[val_idx[0]].copy()
        processed_val_df['target'] = target.iloc[val_idx[0]]
        processed_val_df['split'] = 1
        
        self.data_df = pd.concat([processed_train_df, processed_val_df])

    def load_processed_dataset(self, split) -> Tuple[pd.DataFrame,
                                                     pd.DataFrame,
                                                     pd.DataFrame]:
        if split:
            if SPLIT in self.data_df:
                training_set = self.data_df[self.data_df[SPLIT] == 0]
                val_set = self.data_df[self.data_df[SPLIT] == 1]
            if SPLIT in self.test_df:
                test_set = self.test_df[self.test_df[SPLIT] == 2]
            return training_set, test_set, val_set
        else:
            raise ValueError("The dataset does not have splits, "
                             "load with `split=False`")


def process_categorical_features(df, categorical_features):
    for i in categorical_features:
        df.iloc[:, i].fillna("?", inplace=True) 
        df.iloc[:, i].apply(lambda x: to_float_str(x))
    return df


def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


def process_numerical_features(df, categorical_features):
    columns_to_impute = []
    for i, column in enumerate(df.columns):
        if i not in categorical_features and pd.isnull(df[column]).any():
            columns_to_impute.append(column)
            
    for column_name in columns_to_impute:
        df[column_name + "_imputed"] = pd.isnull(df[column_name]).astype(float)
        df[column_name].fillna(0, inplace=True)

    for i, column in enumerate(df.columns):
        if i not in categorical_features:
            df[column].astype(float, copy=False)
    
    return df
