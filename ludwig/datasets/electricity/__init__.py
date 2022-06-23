#! /usr/bin/env python
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
from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
from ludwig.datasets.mixins.download import UncompressedFileDownloadMixin
from ludwig.datasets.mixins.load import CSVLoadMixin
from ludwig.datasets.mixins.process import IdentityProcessMixin
from ludwig.datasets.registry import register_dataset


def load(cache_dir=DEFAULT_CACHE_LOCATION, split=False):
    dataset = Electricity(cache_dir=cache_dir)
    return dataset.load(split=split)


@register_dataset(name="electricity")
class Electricity(UncompressedFileDownloadMixin, IdentityProcessMixin, CSVLoadMixin, BaseDataset):
    """Electricity demand dataset. Half-hourly electricity demand in Victoria, Australia during 2014, along with
    Melbourne temperatures.

    Source textbook:
    Forecasting: Principles and Practice
        Rob J Hyndman and George Athanasopoulos
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_LOCATION):
        super().__init__(dataset_name="electricity", cache_dir=cache_dir)
