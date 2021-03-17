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
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import  TarDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import *

def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True):
    dataset = AmazonPolarity(cache_dir=cache_dir)
    return dataset.load(split=split)

class AmazonPolarity(TarDownloadMixin, MultifileJoinProcessMixin,
                 CSVLoadMixin, BaseDataset):
    """
        The Amazon Reviews Polarity dataset
        Details:
            34,686,770 Amazon reviews from 6,643,669 users on 2,441,053 \
            products, from the Stanford Network Analysis Project (SNAP). \
            This dataset contains 600,000 training samples and 130,000 \
            testing samples in each class. 
        Dataset source: 
            Character-level Convolutional Networks for Text Classification
            Xiang Zhang et al., 2015
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="amazon_review_polarity", cache_dir=cache_dir)

    def process_downloaded_dataset(self):
        super(AmazonPolarity, self).process_downloaded_dataset(header=None)
        processed_df = pd.read_csv(os.path.join(self.processed_dataset_path,
                                                self.csv_filename))
        processed_df.columns = ['label', 'review_tile', 'review_text', 'split']
        processed_df.to_csv(
            os.path.join(self.processed_dataset_path, self.csv_filename),
            index=False
        )


    
