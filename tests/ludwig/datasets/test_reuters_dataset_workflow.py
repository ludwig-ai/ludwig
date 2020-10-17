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
from unittest.mock import Mock
from pathlib import Path
from ludwig.datasets.reuters.reuters import Reuters


class TestReutersDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self._reuters_handle = Reuters(None)
        self._reuters_handle_mock = Mock(None)
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path,
                                                  "../../../ludwig/datasets/config/dataset_config.yaml")
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self._dataset_version = self._config_file_contents["reuters"]["version"]
        self._processed_data_path = Path.home().joinpath('.ludwig_cache').joinpath("reuters_"
                                                                                   + str(self._dataset_version)) \
            .joinpath('processed.csv')

    def test_download_success(self):
        self._reuters_handle_mock.download()
        self._reuters_handle_mock.download.assert_called_once()

    def test_process_success(self):
        self._reuters_handle.process()
        result = os.path.isfile(self._processed_data_path)
        assert result, True

    def test_load_success(self):
        self._reuters_handle.process()
        transformed_data = self._reuters_handle.load()
        first_key = "2 NEW YORK BANK DISCOUNT WINDOW BORROWINGS 64 MLN DLRS IN FEB 25 WEEK Blah blah blah 3  "
        tmp = transformed_data['class'].where(transformed_data['text'] == first_key)
        assert (tmp[16] == 'Neg-')