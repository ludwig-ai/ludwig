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
import logging
import shutil

import pytest

from ludwig.experiment import experiment_cli
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


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
                'fc_size': 64,
                'num_fc_layers': 5
            },
            'training': {'epochs': 2}
        }

    args = {
        'model_definition': model_definition,
        'skip_save_processed_input': True,
        'skip_save_progress': True,
        'skip_save_unprocessed_output': True,
        'skip_save_model': True,
        'skip_save_log': True
    }
    args.update(kwargs)

    exp_dir_name = experiment_cli(**args)
    shutil.rmtree(exp_dir_name, ignore_errors=True)


@pytest.mark.parametrize(
    'input_test_feature, output_test_feature, output_loss_parameter',
    [
        # numerical features
        (numerical_feature(), numerical_feature(), None),
        (
                numerical_feature(normalization='minmax'),
                numerical_feature(),
                {'loss': {'type': 'mean_squared_error'}}
        ),
        (
                numerical_feature(normalization='zscore'),
                numerical_feature(),
                {'loss': {'type': 'mean_absolute_error'}}
        ),

        # binary feature
        (binary_feature(), binary_feature(), None),

        # Categorical feature
        (category_feature(), category_feature(), None),
        (
                category_feature(),
                category_feature(),
                {'loss': {'type': 'softmax_cross_entropy'}}
        ),
        (
                category_feature(),
                category_feature(),
                {'loss': {
                    'type': 'sampled_softmax_cross_entropy',
                    'sampler': 'fixed_unigram',
                    'negative_samples': 10
                }
                }
        ),
        (
                category_feature(),
                category_feature(),
                {'loss': {
                    'type': 'sampled_softmax_cross_entropy',
                    'sampler': 'uniform',
                    'negative_samples': 10
                }
                }
        ),
        (
                category_feature(),
                category_feature(),
                {'loss': {
                    'type': 'sampled_softmax_cross_entropy',
                    'sampler': 'log_uniform',
                    'negative_samples': 10
                }
                }
        ),
        (
                category_feature(),
                category_feature(),
                {'loss': {
                    'type': 'sampled_softmax_cross_entropy',
                    'sampler': 'learned_unigram',
                    'negative_samples': 10
                }
                }
        )
    ]
)
def test_feature(input_test_feature, output_test_feature,
                 output_loss_parameter, csv_filename):
    input_features = [
        input_test_feature
    ]

    of_test_feature = output_test_feature
    if output_loss_parameter is not None:
        of_test_feature.update(output_loss_parameter)
    output_features = [of_test_feature]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename,
                             1001)

    run_experiment(input_features, output_features, data_csv=rel_path)
