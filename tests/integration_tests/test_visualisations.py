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
import subprocess
import glob
import json

from ludwig.experiment import experiment
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import text_feature, categorical_feature

# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import *


def run_experiment(input_features, output_features, **kwargs):
    """
    Helper method to avoid code repetition in running an experiment. Deletes
    the data saved to disk after running the experiment
    :param input_features: list of input feature dictionaries
    :param output_features: list of output feature dictionaries
    **kwargs you may also pass extra parameters to the experiment as keyword
    arguments
    :return: None
    """
    model_definition = None
    if input_features is not None and output_features is not None:
        # This if is necessary so that the caller can call with
        # model_definition_file (and not model_definition)
        model_definition = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {
                'type': 'concat',
                'fc_size': 14
            },
            'training': {'epochs': 2}
        }

    args = {
        'model_definition': model_definition,
        'skip_save_processed_input': False,
        'skip_save_progress': False,
        'skip_save_unprocessed_output': False,
    }
    args.update(kwargs)

    exp_dir_name = experiment(**args)

    return exp_dir_name

def test_visualisation_learning_curves_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)

    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    train_stats = exp_dir_name + '/training_statistics.json'
    test_cmd_pdf = ['python', '-m', 'ludwig.visualize', '--visualization',
                'learning_curves', '--training_statistics', train_stats,
                '-od', exp_dir_name]
    test_cmd_png = ['python', '-m', 'ludwig.visualize', '--visualization',
                'learning_curves', '--training_statistics', train_stats,
                '-od', exp_dir_name, '-ff', 'png']
    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 5 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)