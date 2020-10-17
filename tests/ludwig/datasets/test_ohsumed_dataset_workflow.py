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
import unittest
import yaml
from pathlib import Path
from unittest.mock import Mock
from ludwig.datasets.ohsumed.ohsumed import OhsuMed


class TestOhsumedDatasetWorkflow(unittest.TestCase):

    def setUp(self):
        self._ohsu_med_handle = OhsuMed(None)
        self._ohsu_med_handle_mock = Mock(None)
        self._initial_path = os.path.abspath(os.path.dirname(__file__))
        self._config_file_location = os.path.join(self._initial_path,
                                                  "../../../ludwig/datasets/config/dataset_config.yaml")
        with open(self._config_file_location) as config_file:
            self._config_file_contents = yaml.load(config_file, Loader=yaml.FullLoader)
        self._dataset_version = self._config_file_contents["ohsumed"]["version"]
        self._processed_data_path = Path.home().joinpath('.ludwig_cache').joinpath("ohsumed_"
                                                                                   + str(self._dataset_version))\
            .joinpath('processed.csv')

    def test_download_success(self):
        self._ohsu_med_handle_mock.download()
        self._ohsu_med_handle_mock.download.assert_called_once()

    def test_process_success(self):
        self._ohsu_med_handle.process()
        result = os.path.isfile(self._processed_data_path)
        assert result

    def test_load_success(self):
        self._ohsu_med_handle.process()
        transformed_data = self._ohsu_med_handle.load()
        # we test a random assortment of keys
        first_key = "Laparoscopic treatment of perforated peptic ulcer. Mouret P  Francois Y  Vignal J  Barth X  Lombard-Platet R."
        second_key = "Cuff size and blood pressure  letter  comment  Gollin S."
        tmp = transformed_data['class'].where(transformed_data['text'] == first_key)
        tmp1 = transformed_data['class'].where(transformed_data['text'] == second_key)
        assert tmp[0] == 'Neg-'
        assert tmp1[16] == 'Neg-'
