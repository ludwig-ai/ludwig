# -*- coding: utf-8 -*-
# Copyright (c) 2020 Uber Technologies, Inc.
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
import contextlib

import pytest
import tensorflow as tf
import tensorflow_addons as tfa

from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import \
    generate_output_features_with_dependencies
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import run_experiment
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import text_feature


@contextlib.contextmanager
def graph_mode():
    prev_mode = tf.config.experimental_functions_run_eagerly()
    try:
        tf.config.experimental_run_functions_eagerly(False)
        yield
    finally:
        tf.config.experimental_run_functions_eagerly(prev_mode)


@pytest.mark.parametrize(
    'output_features',
    [
        # baseline test case
        [
            category_feature(vocab_size=2, reduce_input='sum'),
            sequence_feature(vocab_size=10, max_len=5),
            numerical_feature()
        ],

        # use generator as decoder
        [
            category_feature(vocab_size=2, reduce_input='sum'),
            sequence_feature(vocab_size=10, max_len=5, decoder='generator'),
            numerical_feature()
        ],

        # Generator decoder and reduce_input = None
        [
            category_feature(vocab_size=2, reduce_input='sum'),
            sequence_feature(max_len=5, decoder='generator',
                             reduce_input=None),
            numerical_feature(normalization='minmax')
        ],

        # output features with dependencies single dependency
        generate_output_features_with_dependencies('feat3', ['feat1']),

        # output features with dependencies multiple dependencies
        generate_output_features_with_dependencies('feat3', ['feat1', 'feat2'])
    ]
)
def test_experiment_multiple_seq_seq(csv_filename, output_features):
    with graph_mode():
        input_features = [
            text_feature(vocab_size=100, min_len=1, encoder='stacked_cnn'),
            numerical_feature(normalization='zscore'),
            category_feature(vocab_size=10, embedding_size=5),
            set_feature(),
            sequence_feature(vocab_size=10, max_len=10, encoder='embed')
        ]
        output_features = output_features

        rel_path = generate_data(input_features, output_features, csv_filename)
        run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize('dec_beam_width', [3])
@pytest.mark.parametrize('dec_attention', ['bahdanau'])
@pytest.mark.parametrize('dec_cell_type', ['lstm'])
@pytest.mark.parametrize('enc_cell_type', ['lstm'])
@pytest.mark.parametrize('enc_encoder', ['embed'])
def test_sequence_generator(
        enc_encoder,
        enc_cell_type,
        dec_cell_type,
        dec_attention,
        dec_beam_width,
        csv_filename
):
    tfa.options.TF_ADDONS_PY_OPS = True

    with graph_mode():
        # Define input and output features
        input_features = [
            sequence_feature(
                min_len=5,
                max_len=10,
                encoder='rnn',
                cell_type='lstm',
                reduce_output=None
            )
        ]
        output_features = [
            sequence_feature(
                min_len=5,
                max_len=10,
                decoder='generator',
                cell_type='lstm',
                attention='bahdanau',
                reduce_input=None
            )
        ]

        # Generate test data
        rel_path = generate_data(input_features, output_features, csv_filename)

        # setup encoder specification
        input_features[0]['encoder'] = enc_encoder
        input_features[0]['cell_type'] = enc_cell_type

        # setup decoder specification
        output_features[0]['cell_type'] = dec_cell_type
        output_features[0]['attention'] = dec_attention
        output_features[0]['beam_width'] = dec_beam_width

        # run the experiment
        run_experiment(input_features, output_features, dataset=rel_path)
