#! /usr/bin/env python
# coding=utf-8
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

import pytest
from jsonschema.exceptions import ValidationError
from ludwig.utils.defaults import merge_with_defaults

from ludwig.utils.schema import validate_config, OUTPUT_FEATURE_TYPES

from tests.integration_tests.utils import ENCODERS, numerical_feature, \
    binary_feature, audio_feature, bag_feature, date_feature, h3_feature, \
    set_feature, text_feature, timeseries_feature, vector_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import sequence_feature


def test_config_features():
    all_input_features = [
        audio_feature('/tmp/destination_folder'),
        bag_feature(),
        binary_feature(),
        category_feature(),
        date_feature(),
        h3_feature(),
        image_feature('/tmp/destination_folder'),
        numerical_feature(),
        sequence_feature(),
        set_feature(),
        text_feature(),
        timeseries_feature(),
        vector_feature(),
    ]
    all_output_features = [
        binary_feature(),
        category_feature(),
        numerical_feature(),
        sequence_feature(),
        set_feature(),
        text_feature(),
        vector_feature(),
    ]

    # validate config with all features
    config = {
        'input_features': all_input_features,
        'output_features': all_output_features,
    }
    validate_config(config)

    # make sure all defaults provided also registers as valid
    config = merge_with_defaults(config)
    validate_config(config)

    # test various invalid output features
    input_only_features = [
        feature for feature in all_input_features
        if feature['type'] not in OUTPUT_FEATURE_TYPES
    ]
    for input_feature in input_only_features:
        config = {
            'input_features': all_input_features,
            'output_features': all_output_features + [input_feature],
        }

        dtype = input_feature['type']
        with pytest.raises(ValidationError, match=rf"^'{dtype}' is not one of .*"):
            validate_config(config)

def test_config_encoders():
    for encoder in ENCODERS:
        config = {
            'input_features': [
                sequence_feature(reduce_output='sum', encoder=encoder),
                image_feature('/tmp/destination_folder'),
            ],
            'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
            'combiner': {'type': 'concat', 'fc_size': 14},
        }
        validate_config(config)


def test_config_tabnet():
    config = {
        'input_features': [
            category_feature(vocab_size=2, reduce_input='sum'),
            numerical_feature(),
        ],
        'output_features': [binary_feature(weight_regularization=None)],
        'combiner': {
            'type': 'tabnet',
            'size': 24,
            'output_size': 26,
            'sparsity': 0.000001,
            'bn_virtual_divider': 32,
            'bn_momentum': 0.6,
            'num_steps': 5,
            'relaxation_factor': 1.5,
            'use_keras_batch_norm': False,
            'bn_virtual_bs': 512,
        },
        'training': {
            'batch_size': 16384,
            'eval_batch_size': 500000,
            'epochs': 1000,
            'early_stop': 20,
            'learning_rate': 0.02,
            'optimizer': {
                'type': 'adam'
            },
            'decay': True,
            'decay_steps': 20000,
            'decay_rate': 0.9,
            'staircase': True,
            'regularization_lambda': 1,
            'validation_field': 'label',
        }
    }
    validate_config(config)


def test_config_bad_feature_type():
    config = {
        'input_features': [{'name': 'foo', 'type': 'fake'}],
        'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
        'combiner': {'type': 'concat', 'fc_size': 14},
    }

    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)


def test_config_bad_encoder_name():
    config = {
        'input_features': [sequence_feature(reduce_output='sum', encoder='fake')],
        'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
        'combiner': {'type': 'concat', 'fc_size': 14},
    }

    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)


def test_config_bad_preprocessing_param():
    config = {
        'input_features': [
            sequence_feature(reduce_output='sum', encoder='fake'),
            image_feature(
                '/tmp/destination_folder',
                preprocessing={
                     'in_memory': True,
                     'height': 12,
                     'width': 12,
                     'num_channels': 3,
                     'tokenizer': 'space',
                },
            ),
        ],
        'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
        'combiner': {'type': 'concat', 'fc_size': 14},
    }

    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)


def test_config_bad_combiner():
    config = {
        'input_features': [
            category_feature(vocab_size=2, reduce_input='sum'),
            numerical_feature(),
        ],
        'output_features': [binary_feature(weight_regularization=None)],
        'combiner': {
            'type': 'tabnet',
            # 'dropout': False
        }
    }

    # config is valid at this point
    validate_config(config)

    # combiner without type
    del config['combiner']['type']
    with pytest.raises(ValidationError, match=r"^'type' is a required .*"):
        validate_config(config)

    # bad combiner
    config['combiner']['type'] = 'fake'
    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)

    # bad combiner format (list instead of dict)
    config['combiner'] = [{'type': 'tabnet'}]
    with pytest.raises(ValidationError, match=r"^\[\{'type': 'tabnet'\}\] is not of .*"):
        validate_config(config)
    
    config['combiner'] = {
        'type': 'tabtransformer',
        'num_layers': 10,
        'dropout': False,
    }
    with pytest.raises(ValidationError, match=r"^False is not of type.*"):
        validate_config(config)


def test_config_fill_values():
    vector_fill_values = [
        '1.0 0.0 1.04 10.49',
        '1 2 3 4 5'
        '0'
        '1.0'
        ''
    ]
    binary_fill_values = [
        'yes', 'No', '1', 'TRUE', 1
    ]
    for vector_fill_value, binary_fill_value in zip(vector_fill_values, binary_fill_values):
        config = {
            'input_features': [
                vector_feature(preprocessing={'fill_value': vector_fill_value}),
            ],
            'output_features': [
                binary_feature(preprocessing={'fill_value': binary_fill_value})
            ],
        }
        validate_config(config)

    bad_vector_fill_values = [
        'one two three',
        '1,2,3',
        0
    ]
    bad_binary_fill_values = [
        'one', 2, 'maybe'
    ]
    for vector_fill_value, binary_fill_value in zip(
            vector_fill_values[:3] + bad_vector_fill_values,
            bad_binary_fill_values + binary_fill_values[:3]
    ):
        config = {
            'input_features': [
                vector_feature(preprocessing={'fill_value': vector_fill_value}),
            ],
            'output_features': [
                binary_feature(preprocessing={'fill_value': binary_fill_value})
            ],
        }
        with pytest.raises(ValidationError):
            validate_config(config)
