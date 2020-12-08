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
import os
import shutil

import pandas as pd
import pytest

from ludwig.constants import NAME
from ludwig.experiment import experiment_cli

from tests.integration_tests.utils import binary_feature, sequence_feature, \
    set_feature, text_feature, vector_feature
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
    config = None
    if input_features is not None and output_features is not None:
        # This if is necessary so that the caller can call with
        # config_file (and not config)
        config = {
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
        'config': config,
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

    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize(
    'input_test_feature, output_test_feature',
    [
        ([category_feature()],
         [binary_feature(), binary_feature()]),
        ([category_feature()],
         [category_feature(vocab_size=5), category_feature(vocab_size=7)]),
        ([category_feature()],
         [numerical_feature(), numerical_feature()]),
        ([category_feature()],
         [sequence_feature(vocab_size=5), sequence_feature(vocab_size=7)]),
        ([category_feature()],
         [set_feature(vocab_size=5), set_feature(vocab_size=7)]),
        ([category_feature()],
         [text_feature(vocab_size=5), text_feature(vocab_size=7)]),
        ([category_feature()],
         [vector_feature(), vector_feature()]),
    ]
)
def test_feature_multiple_outputs(input_test_feature, output_test_feature,
                                  csv_filename):
    # Generate test data
    rel_path = generate_data(input_test_feature, output_test_feature,
                             csv_filename, 1001)

    run_experiment(input_test_feature, output_test_feature, dataset=rel_path)


def test_category_int_dtype(tmpdir):
    feature = category_feature()
    input_features = [feature]
    output_features = [binary_feature()]

    csv_fname = generate_data(input_features, output_features,
                              os.path.join(tmpdir, 'dataset.csv'))
    df = pd.read_csv(csv_fname)

    distinct_values = df[feature[NAME]].drop_duplicates().values
    value_map = {v: idx for idx, v in enumerate(distinct_values)}
    df[feature[NAME]] = df[feature[NAME]].map(
        lambda x: value_map[x]
    )

    run_experiment(input_features, output_features, dataset=df)
