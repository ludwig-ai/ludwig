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
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from ludwig.backend import initialize_backend
from ludwig.constants import COLUMN, NAME, PROC_COLUMN
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.features.number_feature import NumberFeatureMixin, numeric_transformation_registry
from ludwig.utils.types import DataFrame


def number_feature():
    feature = {NAME: "x", COLUMN: "x", "type": "number"}
    feature[PROC_COLUMN] = compute_feature_hash(feature)
    return feature


def get_test_data(backend: str) -> Tuple[DataFrame, DataFrame]:
    """Returns test data for the given backend."""
    data_df = pd.DataFrame(pd.Series([2, 4, 6, 8, 10]), columns=["x"])
    proc_df = pd.DataFrame(columns=["x"])
    if backend == "ray":
        import dask.dataframe as dd

        data_df = dd.from_pandas(data_df, npartitions=1).reset_index()
        proc_df = dd.from_pandas(proc_df, npartitions=1).reset_index()
    return data_df, proc_df


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_norm(backend, ray_cluster_2cpu):
    data_df, proc_df = get_test_data(backend)
    backend = initialize_backend(backend)

    feature_1_meta = NumberFeatureMixin.get_feature_meta(data_df["x"], {"normalization": "zscore"}, backend, True)
    feature_2_meta = NumberFeatureMixin.get_feature_meta(data_df["x"], {"normalization": "minmax"}, backend, True)
    feature_3_meta = NumberFeatureMixin.get_feature_meta(data_df["x"], {"normalization": "iq"}, backend, True)

    assert feature_1_meta["mean"] == 6
    assert feature_2_meta["min"] == 2
    assert feature_2_meta["max"] == 10
    assert feature_3_meta["q1"] == 4
    assert feature_3_meta["q2"] == 6
    assert feature_3_meta["q3"] == 8

    # value checks after normalization
    num_feature = number_feature()

    NumberFeatureMixin.add_feature_data(
        feature_config=num_feature,
        input_df=data_df,
        proc_df=proc_df,
        metadata={num_feature[NAME]: feature_1_meta},
        preprocessing_parameters={"normalization": "zscore"},
        backend=backend,
        skip_save_processed_input=False,
    )
    assert np.allclose(
        np.array(proc_df[num_feature[PROC_COLUMN]]), np.array([-1.26491106, -0.63245553, 0, 0.63245553, 1.26491106])
    )

    NumberFeatureMixin.add_feature_data(
        feature_config=num_feature,
        input_df=data_df,
        proc_df=proc_df,
        metadata={num_feature[NAME]: feature_2_meta},
        preprocessing_parameters={"normalization": "minmax"},
        backend=backend,
        skip_save_processed_input=False,
    )
    assert np.allclose(np.array(proc_df[num_feature[PROC_COLUMN]]), np.array([0, 0.25, 0.5, 0.75, 1]))

    NumberFeatureMixin.add_feature_data(
        feature_config=num_feature,
        input_df=data_df,
        proc_df=proc_df,
        metadata={num_feature[NAME]: feature_3_meta},
        preprocessing_parameters={"normalization": "iq"},
        backend=backend,
        skip_save_processed_input=False,
    )
    assert np.allclose(np.array(proc_df[num_feature[PROC_COLUMN]]), np.array([-1, -0.5, 0, 0.5, 1]))


@pytest.mark.parametrize("transformation", numeric_transformation_registry.keys())
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_numeric_transformation_registry(transformation, backend, ray_cluster_2cpu):
    data_df, proc_df = get_test_data(backend)
    backend = initialize_backend(backend)

    feature_meta = NumberFeatureMixin.get_feature_meta(data_df["x"], {"normalization": transformation}, backend, True)

    num_feature = number_feature()

    NumberFeatureMixin.add_feature_data(
        feature_config=num_feature,
        input_df=data_df,
        proc_df=proc_df,
        metadata={num_feature[NAME]: feature_meta},
        preprocessing_parameters={"normalization": transformation},
        backend=backend,
        skip_save_processed_input=False,
    )
