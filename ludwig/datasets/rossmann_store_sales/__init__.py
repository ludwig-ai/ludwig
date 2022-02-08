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
import calendar
import os

import numpy as np
import pandas as pd

from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.kaggle import KaggleDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False, kaggle_username=None, kaggle_key=None):
    dataset = RossmannStoreSales(cache_dir=cache_dir, kaggle_username=kaggle_username, kaggle_key=kaggle_key)
    return dataset.load(split=split)


@register_dataset(name="rossmann_store_sales")
class RossmannStoreSales(CSVLoadMixin, KaggleDownloadMixin, BaseDataset):
    """The Rossmann Store Sales dataset.

    This pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming training data into a destination
    dataframe that can fit into Ludwig's training API.

    Using the time split from the catboost benchmark
    https://github.com/catboost/benchmarks/tree/master/kaggle/rossmann-store-sales
    that is used in the TabNet paper,
    because the test set does not contain sales ground truth
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION, kaggle_username=None, kaggle_key=None):
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        self.is_kaggle_competition = True
        super().__init__(dataset_name="rossmann_store_sales", cache_dir=cache_dir)

    def process_downloaded_dataset(self):

        stores_df = pd.read_csv(os.path.join(self.raw_dataset_path, "store.csv"))

        train_df = pd.read_csv(os.path.join(self.raw_dataset_path, "train.csv"), low_memory=False)
        train_df = preprocess_df(train_df, stores_df)

        train_df["split"] = -1
        train_df.loc[train_df["Year"] == 2014, "split"] = 0
        train_df.loc[train_df["Year"] == 2015, "split"] = 2
        train_df.drop(train_df[train_df["split"] == -1].index, inplace=True)
        df = train_df

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_csv(os.path.join(self.processed_temp_path, self.csv_filename), index=False)
        rename(self.processed_temp_path, self.processed_dataset_path)


def preprocess_dates(df):
    # Make integer Year,Month,Day columns instead of Date
    dates = np.array([[int(v) for v in s.split("-")] for s in df["Date"]])
    df = df.drop(["Date"], axis=1)
    df["Year"] = dates[:, 0]
    df["Month"] = dates[:, 1]
    df["Day"] = dates[:, 2]
    return df


month_abbrs = calendar.month_abbr[1:]
month_abbrs[8] = "Sept"


def preprocess_stores(df, stores_df):
    # join data in df with stores df
    df = df.join(stores_df, on="Store", rsuffix="_right")
    df = df.drop(["Store_right"], axis=1)

    promo2_start_months = [(s.split(",") if not pd.isnull(s) else []) for s in df["PromoInterval"]]

    for month_abbr in month_abbrs:
        df["Promo2Start_" + month_abbr] = np.array(
            [(1 if month_abbr in s else 0) for s in promo2_start_months], dtype=np.int8
        )
    df = df.drop(["PromoInterval"], axis=1)

    return df


int_columns = [
    "Store",
    "DayOfWeek",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
]


def preprocess_df(df, stores_df):
    df = preprocess_dates(df)
    df = preprocess_stores(df, stores_df)

    for column in int_columns:
        df[column] = pd.to_numeric(df[column].fillna(0), downcast="integer")

    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df.loc[df["StateHoliday"] == "0", "StateHoliday"] = "No"

    return df
