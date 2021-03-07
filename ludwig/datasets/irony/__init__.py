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
from ludwig.datasets.mixins.process import IdentityProcessMixin


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Irony(cache_dir=cache_dir)
    return dataset.load(split=split)

class Irony(UncompressedFileDownloadMixin, IdentityProcessMixin,
            CSVLoadMixin, BaseDataset):
    """The Reddit Irony dataset.

    Source Paper: 
    Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)
        Byron C Wallace, Do Kook Choe, Laura Kertz, and Eugene Charniak
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="irony", cache_dir=cache_dir)