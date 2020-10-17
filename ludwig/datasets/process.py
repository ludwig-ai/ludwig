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

"""A mixin to convert a raw csv file into a processed dictionary which itself
is also a csv file, the dictionary should be ready to be absorbed into a destination
dataframe."""


class ProcessMixin:

    """This method currently transforms the raw data into a dictionary which is
    ready to be ingested into a destination dataframe
    args:
        None
    ret:
        None"""
    def process_downloaded_dataset(self):
        dict_reader = csv.DictReader(open(self._raw_file_name))
        value_to_store = None
        for row in dict_reader:
            for key, value in row.items():
                if key == "class":
                    value_to_store = value
                else:
                    key_to_store = value
                    self._result_dict[key_to_store] = value_to_store
        try:
            with open(self._processed_file_name, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in self._result_dict.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

    """A pre-op check to see if the raw or processed file exists as a step to performing
        the next step in the workflow.
        :arg
           file_path (str): the full path to the file to search for
        :return 
            True or false whether we can start the loading into a dataframe"""
    def check_file_existence(self, file_path) -> bool:
        return os.path.isfile(file_path)
