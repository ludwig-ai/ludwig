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
from functools import partial
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import NAME, TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    init_backend,
    RAY_BACKEND_CONFIG,
)


@pytest.mark.distributed
@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.parametrize("distinct_values", [(False, True), ("No", "Yes")])
def test_binary_predictions(tmpdir, backend, distinct_values):
    input_features = [
        category_feature(vocab_size=3),
    ]

    feature = binary_feature()
    output_features = [
        feature,
    ]

    data_csv_path = generate_data(
        input_features,
        output_features,
        os.path.join(tmpdir, "dataset.csv"),
        num_examples=100,
    )
    data_df = pd.read_csv(data_csv_path)

    # Optionally convert bool values to strings, e.g., {'Yes', 'No'}
    false_value, true_value = distinct_values
    data_df[feature[NAME]] = data_df[feature[NAME]].map(lambda x: true_value if x else false_value)
    data_df.to_csv(data_csv_path, index=False)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 1}}

    preds_df = predict_with_backend(tmpdir, config, data_csv_path, backend, num_predict_samples=len(data_df))
    cols = set(preds_df.columns)
    assert f"{feature[NAME]}_predictions" in cols
    assert f"{feature[NAME]}_probabilities_{str(false_value)}" in cols
    assert f"{feature[NAME]}_probabilities_{str(true_value)}" in cols
    assert f"{feature[NAME]}_probability" in cols

    for pred, prob_0, prob_1, prob in zip(
        preds_df[f"{feature[NAME]}_predictions"],
        preds_df[f"{feature[NAME]}_probabilities_{str(false_value)}"],
        preds_df[f"{feature[NAME]}_probabilities_{str(true_value)}"],
        preds_df[f"{feature[NAME]}_probability"],
    ):
        assert pred == false_value or pred == true_value
        if pred == true_value:
            assert prob_1 == prob
        else:
            assert prob_0 == prob
        assert np.allclose(prob_0, 1 - prob_1)


@pytest.mark.distributed
@pytest.mark.parametrize("backend", ["local", "ray"])
@pytest.mark.parametrize("distinct_values", [(0.0, 1.0), (0, 1)])
def test_binary_predictions_with_number_dtype(tmpdir, backend, distinct_values):
    input_features = [
        category_feature(vocab_size=3),
    ]

    feature = binary_feature()
    output_features = [
        feature,
    ]

    data_csv_path = generate_data(
        input_features,
        output_features,
        os.path.join(tmpdir, "dataset.csv"),
        num_examples=100,
    )
    data_df = pd.read_csv(data_csv_path)

    # Optionally convert bool values to strings, e.g., {'Yes', 'No'}
    false_value, true_value = distinct_values
    data_df[feature[NAME]] = data_df[feature[NAME]].map(lambda x: true_value if x else false_value)
    data_df.to_csv(data_csv_path, index=False)

    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 1}}

    preds_df = predict_with_backend(tmpdir, config, data_csv_path, backend, num_predict_samples=len(data_df))
    cols = set(preds_df.columns)
    assert f"{feature[NAME]}_predictions" in cols
    assert f"{feature[NAME]}_probabilities_False" in cols
    assert f"{feature[NAME]}_probabilities_True" in cols
    assert f"{feature[NAME]}_probability" in cols

    for pred, prob_0, prob_1, prob in zip(
        preds_df[f"{feature[NAME]}_predictions"],
        preds_df[f"{feature[NAME]}_probabilities_False"],
        preds_df[f"{feature[NAME]}_probabilities_True"],
        preds_df[f"{feature[NAME]}_probability"],
    ):
        assert isinstance(pred, bool)
        if pred:
            assert prob_1 == prob
        else:
            assert prob_0 == prob
        assert np.allclose(prob_0, 1 - prob_1)


def predict_with_backend(tmpdir, config, data_csv_path, backend, num_predict_samples=None):
    with init_backend(backend):
        if backend == "ray":
            backend = RAY_BACKEND_CONFIG
            backend["processor"]["type"] = "dask"

        ludwig_model = LudwigModel(config, backend=backend)
        _, _, output_directory = ludwig_model.train(
            dataset=data_csv_path,
            output_directory=os.path.join(tmpdir, "output"),
        )
        # Check that metadata JSON saves and loads correctly
        ludwig_model = LudwigModel.load(os.path.join(output_directory, "model"))

        # Produce an even mix of True and False predictions, as the model may be biased towards
        # one direction without training
        def random_logits(*args, num_predict_samples=None, **kwargs):
            return torch.tensor(np.random.uniform(low=-1.0, high=1.0, size=(num_predict_samples,)))

        if num_predict_samples is not None:
            with mock.patch(
                "ludwig.features.binary_feature.BinaryOutputFeature.logits",
                partial(random_logits, num_predict_samples=num_predict_samples),
            ):
                preds_df, _ = ludwig_model.predict(dataset=data_csv_path)
        else:
            preds_df, _ = ludwig_model.predict(dataset=data_csv_path)

    return preds_df
