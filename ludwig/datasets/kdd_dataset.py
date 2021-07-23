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
from zipfile import ZipFile

import fsspec
import pandas as pd
from fsspec.core import split_protocol
from ludwig.datasets.base_dataset import DEFAULT_CACHE_LOCATION, BaseDataset
from ludwig.datasets.mixins.download import UncompressedFileDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import MultifileJoinProcessMixin
from ludwig.utils.fs_utils import makedirs


class KDDCup2009Dataset(UncompressedFileDownloadMixin,
                        MultifileJoinProcessMixin, CSVLoadMixin, BaseDataset):
    """
    The KDD Cup 2009 dataset base class

    Additional Details:

    https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
    """

    def __init__(self, task_name, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="kdd_" + task_name, cache_dir=cache_dir)
        self.task_name = task_name

    def process_downloaded_dataset(self, header=0):
        zip_file = ZipFile(
            os.path.join(self.raw_dataset_path, "orange_small_train.data.zip")
        )
        train_df = pd.read_csv(zip_file.open("orange_small_train.data"),
                               sep='\t')

        zip_file = ZipFile(
            os.path.join(self.raw_dataset_path, "orange_small_test.data.zip")
        )
        test_df = pd.read_csv(zip_file.open("orange_small_test.data"),
                              sep='\t')

        train_df = process_categorical_features(train_df, categorical_features)
        train_df = process_numerical_features(train_df, categorical_features)

        targets = pd.read_csv(
            os.path.join(
                self.raw_dataset_path,
                f"orange_small_train_{self.task_name}.labels"
            ),
            header=None
        )[0].astype(str).apply(lambda x: "true" if x == '1' else "false")

        train_idcs = pd.read_csv(
            os.path.join(
                self.raw_dataset_path,
                f"stratified_train_idx_{self.task_name}.txt"
            ),
            header=None
        )[0]

        val_idcs = pd.read_csv(
            os.path.join(
                self.raw_dataset_path,
                f"stratified_test_idx_{self.task_name}.txt"
            ),
            header=None
        )[0]

        processed_train_df = train_df.iloc[train_idcs].copy()
        processed_train_df['target'] = targets.iloc[train_idcs]
        processed_train_df['split'] = 0

        processed_val_df = train_df.iloc[val_idcs].copy()
        processed_val_df['target'] = targets.iloc[val_idcs]
        processed_val_df['split'] = 1

        test_df['target'] = ''
        test_df['split'] = 2

        df = pd.concat([processed_train_df, processed_val_df, test_df])

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_csv(os.path.join(self.processed_temp_path, self.csv_filename),
                  index=False)

        protocol, _ = split_protocol(self.processed_dataset_path)
        if protocol is not None:
            fs = fsspec.filesystem(protocol)
            fs.copy(self.processed_temp_path,
                    self.processed_dataset_path, recursive=True)
        else:
            os.rename(self.processed_temp_path, self.processed_dataset_path)


def process_categorical_features(df, categorical_features):
    for i in categorical_features:
        df.iloc[:, i].fillna("", inplace=True)
    return df


def process_numerical_features(df, categorical_features):
    for i, column in enumerate(df.columns):
        if i not in categorical_features:
            df[column].astype(float, copy=False)
    return df


categorical_features = {190, 191, 192, 193, 194, 195, 196, 197, 198,
                        199, 200, 201, 202, 203, 204, 205, 206, 207,
                        209, 210, 211, 212, 213, 214, 215, 216, 217,
                        218, 219, 220, 221, 222, 223, 224, 225, 226,
                        227, 228}
