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
from ludwig.data.dataset.pandas import PandasDataset
from ludwig.models.predictor import EXCLUE_PRED_SET


class PartitionedDataset(object):
    def __init__(self, df, features, data_hdf5_fp):
        self.df = df
        self.features = features
        self.data_hdf5_fp = data_hdf5_fp

    def predict_partitions(self, fn, output_features):
        output_columns = []
        for of_name, feature in output_features.items():
            for pred in feature.get_prediction_set():
                if pred not in EXCLUE_PRED_SET:
                    output_columns.append(f'{of_name}_{pred}')

        def wrapped(partition):
            dataset = PandasDataset(partition, self.features, self.data_hdf5_fp)
            predictions = fn(dataset)
            ordered_predictions = predictions[output_columns]
            return ordered_predictions

        return self.df.map_partitions(
            wrapped, meta=[(c, 'object') for c in output_columns]
        )
