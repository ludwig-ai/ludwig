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
import csv
import os


class CSVProcessMixin:
    """A mixin to convert a raw csv file into another processed CSV file."""

    raw_dataset_path: str
    processed_dataset_path: str

    def process_downloaded_dataset(self):
        """
        Transforms the raw data into a dictionary ready to be ingested
        into a destination dataframe.
        """
        result_dict = {}
        with open(self.raw_dataset_path) as f:
            dict_reader = csv.DictReader(f)
            value_to_store = None
            for row in dict_reader:
                for key, value in row.items():
                    if key == "class":
                        value_to_store = value
                    else:
                        key_to_store = value
                        result_dict[key_to_store] = value_to_store

        with open(self.processed_dataset_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in result_dict.items():
                writer.writerow([key, value])

    def is_processed(self) -> bool:
        return os.path.isfile(self.processed_dataset_path)
