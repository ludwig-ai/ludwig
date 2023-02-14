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
import os
import random

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, COLUMN, DROP_ROW, FILL_WITH_MEAN, PREPROCESSING, PROC_COLUMN, TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
    read_csv_with_nan,
    sequence_feature,
    set_feature,
    text_feature,
    vector_feature,
)


def test_missing_value_prediction(tmpdir, csv_filename):
    random.seed(1)
    np.random.seed(1)
    input_features = [
        category_feature(
            encoder={"vocab_size": 2}, reduce_input="sum", preprocessing=dict(missing_value_strategy="fill_with_mode")
        )
    ]
    output_features = [binary_feature()]

    dataset = pd.read_csv(generate_data(input_features, output_features, csv_filename))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
    }
    model = LudwigModel(config)
    _, _, output_dir = model.train(dataset=dataset, output_directory=tmpdir)

    # Set the input column to None, we should be able to replace the missing value with the mode
    # from the training set
    dataset[input_features[0]["name"]] = None
    model.predict(dataset=dataset)

    model = LudwigModel.load(os.path.join(output_dir, "model"))
    model.predict(dataset=dataset)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
def test_missing_values_fill_with_mean(backend, csv_filename, tmpdir, ray_cluster_2cpu):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    kwargs = {PREPROCESSING: {"missing_value_strategy": FILL_WITH_MEAN}}
    input_features = [
        number_feature(**kwargs),
        binary_feature(),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [binary_feature()]
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # run preprocessing
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.preprocess(dataset=training_data_csv_path)


def test_missing_values_drop_rows(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    kwargs = {PREPROCESSING: {"missing_value_strategy": DROP_ROW}}
    input_features = [
        number_feature(),
        binary_feature(),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [
        binary_feature(**kwargs),
        number_feature(**kwargs),
        category_feature(decoder={"vocab_size": 3}, **kwargs),
        sequence_feature(decoder={"vocab_size": 3}, **kwargs),
        text_feature(decoder={"vocab_size": 3}, **kwargs),
        set_feature(decoder={"vocab_size": 3}, **kwargs),
        vector_feature(**kwargs),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    df = read_csv_with_nan(training_data_csv_path, nan_percent=0.1)

    # run preprocessing
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.preprocess(dataset=df)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize("outlier_threshold", [1.0, 3.0])
@pytest.mark.parametrize("outlier_strategy", [None, "fill_with_mean", "fill_with_const"])
def test_outlier_strategy(outlier_strategy, outlier_threshold, backend, tmpdir, ray_cluster_2cpu):
    fill_value = 42
    kwargs = {
        PREPROCESSING: {
            "outlier_strategy": outlier_strategy,
            "outlier_threshold": outlier_threshold,
            "fill_value": fill_value,
        }
    }
    input_features = [
        number_feature(**kwargs),
    ]
    output_features = [binary_feature()]

    # Values that will be 1 and 3 std deviations from the mean, respectively
    sigma1, sigma1_idx = -150, 4
    sigma3, sigma3_idx = 300, 11

    num_col = np.array([77, 24, 29, 29, sigma1, 71, 46, 95, 20, 52, 85, sigma3, 74, 10, 98, 53, 110, 94, 62, 13])
    expected_fill_value = num_col.mean() if outlier_strategy == "fill_with_mean" else fill_value

    input_col = input_features[0][COLUMN]
    output_col = output_features[0][COLUMN]

    bin_col = np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.bool_)
    dataset_df = pd.DataFrame(
        data={
            input_col: num_col,
            output_col: bin_col,
        }
    )

    dataset_fp = os.path.join(tmpdir, "dataset.csv")
    dataset_df.to_csv(dataset_fp)

    config = {
        "input_features": input_features,
        "output_features": output_features,
    }

    # Run preprocessing
    ludwig_model = LudwigModel(config, backend=backend)
    proc_dataset = ludwig_model.preprocess(training_set=dataset_fp)

    # Check preprocessed output
    proc_df = ludwig_model.backend.df_engine.compute(proc_dataset.training_set.to_df())
    proc_col = input_features[0][PROC_COLUMN]

    assert len(proc_df) == len(dataset_df)

    # Check that values over 1 std are replaced
    if outlier_strategy is not None and outlier_threshold <= 1.0:
        assert np.isclose(proc_df[proc_col][sigma1_idx], expected_fill_value)
    else:
        assert np.isclose(proc_df[proc_col][sigma1_idx], dataset_df[input_col][sigma1_idx])

    # Check that values over 3 std are replaced
    if outlier_strategy is not None and outlier_threshold <= 3.0:
        assert np.isclose(proc_df[proc_col][sigma3_idx], expected_fill_value)
    else:
        assert np.isclose(proc_df[proc_col][sigma3_idx], dataset_df[input_col][sigma3_idx])
