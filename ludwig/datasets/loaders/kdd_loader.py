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
from typing import Optional

import pandas as pd

from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader


class KDDCup2009Loader(DatasetLoader):
    def __init__(
        self, config: DatasetConfig, cache_dir: Optional[str] = None, task_name="", include_test_download=False
    ):
        super().__init__(config, cache_dir=cache_dir)
        self.task_name = task_name
        self.include_test_download = include_test_download

    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        """Loads a file into a dataframe."""
        return pd.read_csv(file_path, sep="\t")

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        train_df = super().transform_dataframe(dataframe)
        train_df = process_categorical_features(train_df, categorical_features)
        train_df = process_number_features(train_df, categorical_features)

        targets = (
            pd.read_csv(os.path.join(self.raw_dataset_dir, f"orange_small_train_{self.task_name}.labels"), header=None)[
                0
            ]
            .astype(str)
            .apply(lambda x: "true" if x == "1" else "false")
        )

        train_idcs = pd.read_csv(
            os.path.join(self.raw_dataset_dir, f"stratified_train_idx_{self.task_name}.txt"), header=None
        )[0]

        val_idcs = pd.read_csv(
            os.path.join(self.raw_dataset_dir, f"stratified_test_idx_{self.task_name}.txt"), header=None
        )[0]

        processed_train_df = train_df.iloc[train_idcs].copy()
        processed_train_df["target"] = targets.iloc[train_idcs]
        processed_train_df["split"] = 0

        processed_val_df = train_df.iloc[val_idcs].copy()
        processed_val_df["target"] = targets.iloc[val_idcs]
        processed_val_df["split"] = 1

        if self.include_test_download:
            test_df = self.load_file_to_dataframe(os.path.join(self.raw_dataset_dir, "orange_small_test.data"))
            test_df["target"] = ""  # no ground truth labels for test download
            test_df["split"] = 2
            df = pd.concat([processed_train_df, processed_val_df, test_df])
        else:
            df = pd.concat([processed_train_df, processed_val_df])

        return df


def process_categorical_features(df, categorical_features):
    for i in categorical_features:
        df.iloc[:, i].fillna("", inplace=True)
    return df


def process_number_features(df, categorical_features):
    for i, column in enumerate(df.columns):
        if i not in categorical_features:
            df[column].astype(float, copy=False)
    return df


categorical_features = {
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
}


class KDDAppetencyLoader(KDDCup2009Loader):
    """The KDD Cup 2009 Appetency dataset.

    https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, include_test_download=False):
        super().__init__(
            config, cache_dir=cache_dir, task_name="appetency", include_test_download=include_test_download
        )


class KDDChurnLoader(KDDCup2009Loader):
    """The KDD Cup 2009 Churn dataset.

    https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, include_test_download=False):
        super().__init__(config, cache_dir=cache_dir, task_name="churn", include_test_download=include_test_download)


class KDDUpsellingLoader(KDDCup2009Loader):
    """The KDD Cup 2009 Upselling dataset.

    https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None, include_test_download=False):
        super().__init__(
            config, cache_dir=cache_dir, task_name="upselling", include_test_download=include_test_download
        )
