#! /usr/bin/env python
# coding=utf-8
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
from ludwig.datasets.mixins.kaggle import KaggleMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Titanic(cache_dir=cache_dir)
    return dataset.load(split=split)


class Titanic(IdentityProcessMixin, KaggleMixin, CSVLoadMixin, BaseDataset):
    """The Titanic dataset.

    This dataset is constructed using the kaggle API.
    For a detailed description of the dataset see here: https://www.kaggle.com/c/titanic-dataset/data

    Like some of the other datasets this dataset is broken up into a titanic_train.csv
    and a titanic_test.csv.

    Here is a sample dataset structure:

    passenger_id,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,home.dest,survived
    1216,3,"Smyth, Miss. Julia",female,,0,0,335432,7.7333,,Q,13,,,1
    699,3,"Cacic, Mr. Luka",male,38.0,0,0,315089,8.6625,,S,,,Croatia,0
    1267,3,"Van Impe, Mrs. Jean Baptiste (Rosalie Paula Govaert)",female,30.0,1,1,345773,24.15,,S,,,,0
    449,2,"Hocking, Mrs. Elizabeth (Eliza Needs)",female,54.0,1,3,29105,23.0,,S,4,,"Cornwall / Akron, OH",1
    576,2,"Veal, Mr. James",male,40.0,0,0,28221,13.0,,S,,,"Barre, Co Washington, VT",0

    In the construction of this dataset, we again combine the train and the test datasets into 1 dataset
    and add a split column to identify whether the dataset is a train or test dataset

    This class pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming
    training data into a destination dataframe that can be use by Ludwig.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name='titanic', cache_dir=cache_dir)

    def output_training_and_test_data(self):
        """The final method where we create a training and test file by iterating through
        both of these files
        """
        train_df = pd.read_csv(os.path.join(self.processed_temp_path, "titanic_train.csv"))
        test_df = pd.read_csv(os.path.join(self.processed_temp_path, "titanic_test.csv"))
        train_df["split"] = 0
        test_df["split"] = 2
        final_df = pd.concat([train_df, train_df], axis=1)
        final_df.to_csv(self.processed_dataset_path)



