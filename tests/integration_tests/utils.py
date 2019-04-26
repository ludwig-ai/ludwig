# -*- coding: utf-8 -*-
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
from string import Template

import pandas as pd
import yaml

from ludwig.data.dataset_synthesyzer import build_synthetic_dataset

ENCODERS = ['embed', 'rnn', 'parallel_cnn', 'cnnrnn', 'stacked_parallel_cnn',
            'stacked_cnn']

model_definition_template = Template(
    '{input_features: ${input_name}, output_features: ${output_name}, '
    'training: {epochs: 2}, combiner: {type: concat, fc_size: 56}}')


def generate_data(input_features,
                  output_features,
                  filename='test_csv.csv',
                  num_examples=25):
    """
    Helper method to generate synthetic data based on input, output feature
    specs
    :param num_examples: number of examples to generate
    :param input_features: schema
    :param output_features: schema
    :param filename: path to the file where data is stored
    :return:
    """
    features = yaml.safe_load(input_features) + yaml.safe_load(
        output_features)
    df = build_synthetic_dataset(num_examples, features)
    data = [next(df) for _ in range(num_examples)]

    dataframe = pd.DataFrame(data[1:], columns=data[0])
    dataframe.to_csv(filename, index=False)

    return filename
