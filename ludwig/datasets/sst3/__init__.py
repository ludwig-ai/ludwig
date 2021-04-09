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
from ludwig.datasets.base_dataset import DEFAULT_CACHE_LOCATION
from ludwig.datasets.sst2 import SST


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False,
         include_subtrees=False, convert_parentheses=True):
    dataset = SST3(cache_dir=cache_dir, include_subtrees=include_subtrees,
                   convert_parentheses=convert_parentheses)
    return dataset.load(split=split)


class SST3(SST):
    """The SST5 dataset.

    This dataset is constructed using the Stanford Sentiment Treebank Dataset.
    This dataset contains five labels (very negative, negative, neutral, 
    positive, very positive) for each sample.

    In the original dataset, the  5 labels: very negative, negative, neutral, positive, 
    and very positive have the following cutoffs:
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]

    This class pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming
    training data into a destination dataframe that can be use by Ludwig.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION,
                 include_subtrees=False,
                 convert_parentheses=True,
                 remove_duplicates=False):
        super().__init__(dataset_name='sst3', cache_dir=cache_dir,
                         include_subtrees=include_subtrees,
                         convert_parentheses=convert_parentheses,
                         remove_duplicates=False)

    def get_sentiment_label(self, id2sent, phrase_id):
        sentiment = id2sent[phrase_id]
        if sentiment <= 0.4:
            return 'negative'
        elif sentiment <= 0.6:
            return 'neutral'
        elif sentiment <= 1.0:
            return 'positive'
        return 'neutral'
