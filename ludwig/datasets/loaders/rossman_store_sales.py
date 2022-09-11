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
import calendar
import os
from typing import List

import numpy as np
import pandas as pd

from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class RossmanStoreSalesLoader(DatasetLoader):
    """The Rossmann Store Sales dataset."""

    def load_unprocessed_dataframe(self, file_paths: List[str]) -> pd.DataFrame:
        """Load dataset files into a dataframe."""

        stores_df = pd.read_csv(os.path.join(self.raw_dataset_dir, "store.csv"))

        train_df = pd.read_csv(os.path.join(self.raw_dataset_dir, "train.csv"), low_memory=False)
        train_df = preprocess_df(train_df, stores_df)

        train_df["split"] = -1
        train_df.loc[train_df["Year"] == 2014, "split"] = 0
        train_df.loc[train_df["Year"] == 2015, "split"] = 2
        train_df.drop(train_df[train_df["split"] == -1].index, inplace=True)
        return train_df


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
