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
from ludwig.datasets.registry import register_dataset
from ludwig.utils.fs_utils import makedirs, rename


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = AdultCensusIncome(cache_dir=cache_dir)
    return dataset.load(split=split)


@register_dataset(name="adult_census_income")
class AdultCensusIncome(UncompressedFileDownloadMixin, CSVLoadMixin, BaseDataset):
    """The Adult Census Income dataset.

    Predict whether income exceeds $50K/yr based on census data.

        More info:
        https://archive.ics.uci.edu/ml/datasets/adult
    """

    raw_dataset_path: str
    processed_dataset_path: str

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="adult_census_income", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        train_df = pd.read_csv(os.path.join(self.raw_dataset_path, "adult.data"), header=None)
        test_df = pd.read_csv(os.path.join(self.raw_dataset_path, "adult.test"), header=None, skiprows=1)

        # age: continuous.
        # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. # noqa: E501
        # fnlwgt: continuous.
        # education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. # noqa: E501
        # education-num: continuous.
        # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.    # noqa: E501
        # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. # noqa: E501
        # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        # sex: Female, Male.
        # capital-gain: continuous.
        # capital-loss: continuous.
        # hours-per-week: continuous.
        # native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.   # noqa: E501
        # income: >50K, <=50K.
        columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        train_df.columns = columns
        test_df.columns = columns
        # Remove the trailing period on the income field in adult.test (not in adult.data)
        test_df["income"] = test_df["income"].str.rstrip(".")

        train_df["split"] = 0
        test_df["split"] = 2

        df = pd.concat([train_df, test_df])

        makedirs(self.processed_temp_path, exist_ok=True)
        df.to_csv(os.path.join(self.processed_temp_path, self.csv_filename), index=False)

        rename(self.processed_temp_path, self.processed_dataset_path)
