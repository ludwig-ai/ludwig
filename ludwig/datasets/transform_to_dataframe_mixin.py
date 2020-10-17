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

"""A class whose responsibility it is to take in a csv file and convert it into
any type of destination dataframe"""

import pandas as pd


class TransformToDataframeMixin:

    def __init__(self, dataset_name, cache_location):
        super().__init__(dataset_name, cache_location)

    def transform_processed_data_to_dataframe(self) -> pd.DataFrame:
        column_names = ["text", "class"]
        if self.check_file_existence(self._processed_file_name):
            return pd.read_csv(self._processed_file_name, names=column_names)
        else:
            self.download()
            self.process()
            self.load()
