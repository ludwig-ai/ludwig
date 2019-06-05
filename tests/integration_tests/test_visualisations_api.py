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

import shutil
import glob

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import read_csv
from ludwig import visualize
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import categorical_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature
from unittest import mock

# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import csv_filename


def run_api_experiment(input_features, output_features):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)

    # # Training with csv
    # model.train(
    #     data_csv=data_csv,
    #     skip_save_processed_input=False,
    #     skip_save_progress=False,
    #     skip_save_unprocessed_output=False
    # )
    #
    # model.predict(data_csv=data_csv)
    #
    # # Remove results/intermediate data saved to disk
    # shutil.rmtree(model.exp_dir_name, ignore_errors=True)

    # Training with dataframe
    return model

def test_api_intent_classification(csv_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]
    encoder = 'cnnrnn'

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    data_df = read_csv(data_csv)
    train_stats = model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    vis_output_pattern_pdf = model.exp_dir_name + '/*.pdf'
    visualize.learning_curves_api(train_stats, field=None, output_directory=model.exp_dir_name)
    figure_cnt = glob.glob(vis_output_pattern_pdf)
    assert 5 == len(figure_cnt)
    model.close()
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)


