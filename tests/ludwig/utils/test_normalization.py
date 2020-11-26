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
import numpy as np
import pandas as pd

from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import PROC_COLUMN, COLUMN, NAME
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.features.numerical_feature import NumericalFeatureMixin


def numerical_feature():
    feature = {NAME: 'x', COLUMN: 'x', 'type': 'numerical'}
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


data_df = pd.DataFrame(pd.Series([
    2,
    4,
    6,
    8,
    10
]), columns=['x'])

proc_df = pd.DataFrame(columns=['x'])


def test_norm():
    feature_1_meta = NumericalFeatureMixin.get_feature_meta(
        data_df['x'], {'normalization': 'zscore'}, LOCAL_BACKEND
    )
    feature_2_meta = NumericalFeatureMixin.get_feature_meta(
        data_df['x'], {'normalization': 'minmax'}, LOCAL_BACKEND
    )

    assert feature_1_meta['mean'] == 6
    assert feature_2_meta['min'] == 2
    assert feature_2_meta['max'] == 10

    # value checks after normalization
    num_feature = numerical_feature()

    NumericalFeatureMixin.add_feature_data(
        feature=num_feature,
        input_df=data_df,
        proc_df=proc_df,
        metadata={num_feature[NAME]: feature_1_meta},
        preprocessing_parameters={'normalization': 'zscore'},
        backend=LOCAL_BACKEND
    )
    assert np.allclose(np.array(proc_df[num_feature[PROC_COLUMN]]),
                       np.array([-1.26491106, -0.63245553, 0, 0.63245553,
                                 1.26491106])
                       )

    NumericalFeatureMixin.add_feature_data(
        feature=num_feature,
        input_df=data_df,
        proc_df=proc_df,
        metadata={num_feature[NAME]: feature_2_meta},
        preprocessing_parameters={'normalization': 'minmax'},
        backend=LOCAL_BACKEND
    )
    assert np.allclose(np.array(proc_df[num_feature[PROC_COLUMN]]),
                       np.array([0, 0.25, 0.5, 0.75, 1])
                       )
