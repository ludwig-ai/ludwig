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

import yaml

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import model_definition_template
# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import csv_filename


def run_api_experiment(input_features, output_features, data_csv):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = model_definition_template.substitute(
        input_name=input_features,
        output_name=output_features
    )

    model = LudwigModel(yaml.load(model_definition))

    # Training with csv
    model.train(
        data_csv=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_csv=data_csv)

    # Training with dataframe
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_df=data_df)


def test_api_intent_classification(csv_filename):
    # Single sequence input, single category output
    input_features = Template('[{name: utterance, type: sequence,'
                              'vocab_size: 10, max_len: 10, '
                              'encoder: ${encoder}, reduce_output: sum}]')
    output_features = "[{name: intent, type: category, vocab_size: 2," \
                      " reduce_input: sum}] "

    # Generate test data
    rel_path = generate_data(
        input_features.substitute(encoder='rnn'), output_features, csv_filename
    )
    for encoder in ENCODERS:
        run_api_experiment(
            input_features.substitute(encoder=encoder),
            output_features,
            data_csv=rel_path
        )
