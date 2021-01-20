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
from ludwig.datasets.mixins.download import UncompressedFileDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import *

def load(cache_dir=DEFAULT_CACHE_LOCATION, split=True):
    dataset = AGNews(cache_dir=cache_dir)
    return dataset.load(split=split)

class AGNews(UncompressedFileDownloadMixin, MultifileJoinProcessMixin,
                 CSVLoadMixin, BaseDataset):
    """The AGNews dataset"""
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="agnews", cache_dir=cache_dir)

    def read_file(self, filetype, filename):
        file_df = pd.read_csv(
                os.path.join(self.raw_dataset_path, filename))
        # class_index : number between 1-4 where 
        # 1-World, 2-Sports, 3-Business, 4-Science/Tech
        file_df.columns = ['class_index', 'title', 'description']
        return file_df


    
