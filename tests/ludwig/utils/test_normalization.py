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
import numpy as np

from ludwig.features.numerical_feature import NumericalBaseFeature

def numerical_feature():
    return {'name': 'x' , 'type': 'numerical'}

data_df = pd.DataFrame(pd.Series([
    2,
    4,
    6,
    8,
    10
]), columns=['x'])

data = pd.DataFrame(pd.Series([
    2,
    4,
    6,
    8,
    10
]), columns=['x'])

feature_1 = NumericalBaseFeature(numerical_feature())
feature_2 = NumericalBaseFeature(numerical_feature())

def test_norm():
    feature_1_meta = feature_1.get_feature_meta(
        data_df['x'], {'normalization': 'zscore'}
    )
    feature_2_meta = feature_1.get_feature_meta(
        data_df['x'], {'normalization': 'minmax'}
    )
    
    assert feature_1_meta['mean'] == 6
    assert feature_2_meta['min'] == 2
    assert feature_2_meta['max'] == 10
    
    # value checks after normalization
    feature_1.add_feature_data(
        feature=numerical_feature(),
        dataset_df=data_df,
        data=data,
        metadata={'x': feature_1_meta},
        preprocessing_parameters={'normalization': 'zscore'}
    )
    assert np.allclose(np.array(data['x']), 
        np.array([-1.26491106, -0.63245553,  0,  0.63245553,  1.26491106])
    )

    feature_2.add_feature_data(
        feature=numerical_feature(),
        dataset_df=data_df,
        data=data,
        metadata={'x': feature_2_meta},
        preprocessing_parameters={'normalization': 'minmax'}
    )
    assert np.allclose(np.array(data['x']), 
        np.array([0, 0.25, 0.5 , 0.75, 1])
    )

