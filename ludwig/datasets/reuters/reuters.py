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
from ludwig.datasets.base_dataset import BaseDataset
from ludwig.datasets.csv_workflow_mixin import CsvWorkflowMixin
from ludwig.datasets.zip_download_workflow_mixin import ZipDownloadWorkflowMixin
from ludwig.datasets.transform_to_dataframe_mixin import TransformToDataframeMixin

"""The ohsumed dataset which pulls in an array of mixins for different types of functionality
which belongs in the workflow for ingesting and transforming training data into a destination
dataframe that can fit into Ludwig's training API"""


class Reuters(CsvWorkflowMixin, ZipDownloadWorkflowMixin, TransformToDataframeMixin, BaseDataset):

    def __init__(self, cache_location):
        super().__init__(dataset_name="reuters", cache_location=cache_location)
