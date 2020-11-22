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

from ludwig.utils.schema import validate_config

from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import sequence_feature


def test_config_basic():
    for encoder in ENCODERS:
        config = {
            'input_features': [sequence_feature(reduce_output='sum', encoder=encoder)],
            'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
            'combiner': {'type': 'concat', 'fc_size': 14},
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
