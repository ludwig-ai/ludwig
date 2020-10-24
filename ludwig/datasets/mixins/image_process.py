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
from ludwig.datasets.mixins.process import IdentityProcessMixin


class ImageProcessMixin(IdentityProcessMixin):
    """A mixin that downloads the mnist dataset and extracts a training and test csv file set"""

    raw_dataset_path: str
    processed_dataset_path: str

    def process_downloaded_dataset(self):
        os.rename(self.raw_dataset_path, self.processed_dataset_path)