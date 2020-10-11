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
import yaml
import unittest
from pathlib import Path
from ludwig.datasets.reuters.reuters import Reuters


class TestReutersDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self._reuters_handle = Reuters()
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path, "../../../ludwig/datasets/text/versions.yaml")
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self._cur_version = self._config_file_contents["text"]["reuters"]

    def test_download_success(self):
        self._reuters_handle.download("reuters")
        download_path = Path.home().joinpath('.ludwig_cache').joinpath("reuters_"
                                                                       + str(self._cur_version)).joinpath('raw.csv')
        result = os.path.isfile(download_path)
        assert (result, True)

    def test_process_success(self):
        self._reuters_handle.process()
        processed_data_path = Path.home().joinpath('.ludwig_cache').joinpath("reuters_"
                                                                             + str(self._cur_version)).joinpath(
            'processed.csv')
        result = os.path.isfile(processed_data_path)
        assert (result, True)

    def test_load_success(self):
        self._reuters_handle.process()
        transformed_data = self._reuters_handle.load()
        first_key = "2 NEW YORK BANK DISCOUNT WINDOW BORROWINGS 64 MLN DLRS IN FEB 25 WEEK Blah blah blah 3  "
        tmp = transformed_data['class'].where(transformed_data['text'] == first_key)
        assert (tmp[16] == 'Neg-')