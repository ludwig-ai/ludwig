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
import json

import pytest
from jsonschema.exceptions import ValidationError

from ludwig.features.audio_feature import AudioFeatureMixin
from ludwig.features.bag_feature import BagFeatureMixin
from ludwig.features.binary_feature import BinaryFeatureMixin
from ludwig.features.category_feature import CategoryFeatureMixin
from ludwig.features.date_feature import DateFeatureMixin
from ludwig.features.h3_feature import H3FeatureMixin
from ludwig.features.image_feature import ImageFeatureMixin
from ludwig.features.numerical_feature import NumericalFeatureMixin
from ludwig.features.sequence_feature import SequenceFeatureMixin
from ludwig.features.set_feature import SetFeatureMixin
from ludwig.features.text_feature import TextFeatureMixin
from ludwig.features.timeseries_feature import TimeseriesFeatureMixin
from ludwig.features.vector_feature import VectorFeatureMixin
from ludwig.utils.defaults import merge_with_defaults

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
        }
    }

    # config is valid at this point
    validate_config(config)

    # combiner without type
    del config['combiner']['type']
    with pytest.raises(ValidationError, match=r"^'type' is a required .*"):
        validate_config(config)

    # bad combiner type
    config['combiner']['type'] = 'fake'
    with pytest.raises(ValidationError, match=r"^'fake' is not one of .*"):
        validate_config(config)

    # bad combiner format (list instead of dict)
    config['combiner'] = [{'type': 'tabnet'}]
    with pytest.raises(ValidationError, match=r"^\[\{'type': 'tabnet'\}\] is not of .*"):
        validate_config(config)
    
    # bad combiner parameter types
    config['combiner'] = {
        'type': 'tabtransformer',
        'num_layers': 10,
        'dropout': False,
    }
    with pytest.raises(ValidationError, match=r"^False is not of type.*"):
        validate_config(config)

    # bad combiner parameter range
    config['combiner'] = {
        'type': 'transformer',
        'dropout': -1,
    }
    with pytest.raises(ValidationError, match=r"less than the minimum.*"):
        validate_config(config)

def test_config_bad_combiner_types_enums():
    config = {
        'input_features': [
            category_feature(vocab_size=2, reduce_input='sum'),
            numerical_feature(),
        ],
        'output_features': [binary_feature(weight_regularization=None)],
        'combiner': {
            'type': 'concat',
            'weights_initializer': 'zeros'
        },
    }

    # config is valid at this point
    validate_config(config)

    # Test weights initializer:
    config['combiner']['weights_initializer'] = {'test': 'fail'}
    with pytest.raises(ValidationError, match=r"{'test': 'fail'} is not of*"):
        validate_config(config)
    config['combiner']['weights_initializer'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not of*"):
        validate_config(config)
    
    # Test bias initializer:
    del config['combiner']['weights_initializer']
    config['combiner']['bias_initializer'] = 'kaiming_uniform'
    validate_config(config)
    config['combiner']['bias_initializer'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not of*"):
        validate_config(config)
    
    # Test weights regularizer:
    del config['combiner']['bias_initializer']
    config['combiner']['weights_regularizer'] = 'l1'
    validate_config(config)
    config['combiner']['weights_regularizer'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)
    
    # Test bias regularizer:
    del config['combiner']['weights_regularizer']
    config['combiner']['bias_regularizer'] = 'l1_l2'
    validate_config(config)
    config['combiner']['bias_regularizer'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)
    
    # Test activity regularizer:
    del config['combiner']['bias_regularizer']
    config['combiner']['activity_regularizer'] = 'l1_l2'
    validate_config(config)
    config['combiner']['activity_regularizer'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)
    
    # Test norm:
    del config['combiner']['activity_regularizer']
    config['combiner']['norm'] = 'batch'
    validate_config(config)
    config['combiner']['norm'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config)
    
    # Test activation:
    del config['combiner']['norm']
    config['combiner']['activation'] = 'relu'
    validate_config(config)
    config['combiner']['activation'] = 123
    with pytest.raises(ValidationError, match=r"123 is not of type*"):
        validate_config(config)
    
    # Test reduce_output:
    del config['combiner']['activation']
    config2 = {**config}
    config2['combiner']['type'] = 'tabtransformer'
    config2['combiner']['reduce_output'] = 'sum'
    validate_config(config)
    config2['combiner']['reduce_output'] = 'fail'
    with pytest.raises(ValidationError, match=r"'fail' is not one of*"):
        validate_config(config2)

    # Test reduce_output = None:
    config2['combiner']['reduce_output'] = None
    validate_config(config2)


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


def test_validate_with_preprocessing_defaults():
    config = {
        "input_features": [
            audio_feature('/tmp/destination_folder',
                          preprocessing=AudioFeatureMixin.preprocessing_defaults),
            bag_feature(preprocessing=BagFeatureMixin.preprocessing_defaults),
            binary_feature(preprocessing=BinaryFeatureMixin.preprocessing_defaults),
            category_feature(preprocessing=CategoryFeatureMixin.preprocessing_defaults),
            date_feature(preprocessing=DateFeatureMixin.preprocessing_defaults),
            h3_feature(preprocessing=H3FeatureMixin.preprocessing_defaults),
            image_feature('/tmp/destination_folder',
                          preprocessing=ImageFeatureMixin.preprocessing_defaults),
            numerical_feature(preprocessing=NumericalFeatureMixin.preprocessing_defaults),
            sequence_feature(preprocessing=SequenceFeatureMixin.preprocessing_defaults),
            set_feature(preprocessing=SetFeatureMixin.preprocessing_defaults),
            text_feature(preprocessing=TextFeatureMixin.preprocessing_defaults),
            timeseries_feature(preprocessing=TimeseriesFeatureMixin.preprocessing_defaults),
            vector_feature(preprocessing=VectorFeatureMixin.preprocessing_defaults),
        ],
        "output_features": [{"name": "target", "type": "category"}],
        "training": {
            "decay": True,
            "learning_rate": 0.001,
            "validation_field": "target",
            "validation_metric": "accuracy"
        },
    }

    validate_config(config)
    config = merge_with_defaults(config)
    validate_config(config)
