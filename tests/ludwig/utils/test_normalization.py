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
import pandas as pd

from ludwig.features.numerical_feature import NumericalBaseFeature

def numerical_feature():
    return {'name': 'norm_test' , 'type': 'numerical'}

column = pd.Series([
    2,
    4,
    6,
    8,
    10
])

feature_1 = NumericalBaseFeature(numerical_feature())
feature_2 = NumericalBaseFeature(numerical_feature())

def test_norm():
    assert feature_1.get_feature_meta(
        column, {'normalization': 'zscore'}
    )['mean'] == 6
    assert feature_2.get_feature_meta(
        column, {'normalization': 'minmax'}
    )['min'] == 2
    assert feature_2.get_feature_meta(
        column, {'normalization': 'minmax'}
    )['max'] == 10

