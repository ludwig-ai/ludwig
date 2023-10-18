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
from ludwig.constants import BATCH_SIZE, DECODER, NAME, TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    RAY_BACKEND_CONFIG,
    set_feature,
    text_feature,
)


def random_binary_logits(*args, num_predict_samples, **kwargs):
    # Produce an even mix of True and False predictions, as the model may be biased
    # towards one direction without training
    return torch.tensor(np.random.uniform(low=-1.0, high=1.0, size=(num_predict_samples,)), dtype=torch.float32)


def random_set_logits(*args, num_predict_samples, vocab_size, pct_positive, **kwargs):
    # Produce a desired mix of predictions based on the pct_positive, as the model may be biased
    # towards one direction without training
    num_positive = int(num_predict_samples * pct_positive)
    num_negative = num_predict_samples - num_positive
    negative_logits = np.random.uniform(low=-1.0, high=-0.1, size=(num_negative, vocab_size))
    positive_logits = np.random.uniform(low=0.1, high=1.0, size=(num_positive, vocab_size))
    logits = np.concatenate([negative_logits, positive_logits], axis=0)
    return torch.tensor(logits, dtype=torch.float32)  # simulate torch model output


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize("distinct_values", [(False, True), ("No", "Yes")])
def test_binary_predictions(tmpdir, backend, distinct_values, ray_cluster_2cpu):
    input_features = [
        category_feature(encoder={"vocab_size": 3}),
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

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 1, BATCH_SIZE: 128},
    }

    patch_args = (
        "ludwig.features.binary_feature.BinaryOutputFeature.logits",
        partial(random_binary_logits, num_predict_samples=len(data_df)),
    )

    preds_df, _ = predict_with_backend(tmpdir, config, data_csv_path, backend, patch_args=patch_args)
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("ray", id="ray", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize("distinct_values", [(0.0, 1.0), (0, 1)])
def test_binary_predictions_with_number_dtype(tmpdir, backend, distinct_values, ray_cluster_2cpu):
    input_features = [
        category_feature(encoder={"vocab_size": 3}),
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

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 1, BATCH_SIZE: 128},
    }

    patch_args = (
        "ludwig.features.binary_feature.BinaryOutputFeature.logits",
        partial(random_binary_logits, num_predict_samples=len(data_df)),
    )

    preds_df, _ = predict_with_backend(tmpdir, config, data_csv_path, backend, patch_args=patch_args)
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


@pytest.mark.parametrize("pct_positive", [1.0, 0.5, 0.0])
def test_set_feature_saving(tmpdir, pct_positive):
    backend = "local"
    input_features = [
        text_feature(encoder={"vocab_size": 3}),
    ]

    feature = set_feature(output_feature=True)
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

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 1, BATCH_SIZE: 128},
    }

    patch_args = (
        "ludwig.features.set_feature.SetOutputFeature.logits",
        partial(
            random_set_logits,
            num_predict_samples=len(data_df),
            vocab_size=feature[DECODER]["vocab_size"] + 1,  # +1 for UNK
            pct_positive=pct_positive,
        ),
    )

    preds_df, ludwig_model = predict_with_backend(tmpdir, config, data_csv_path, backend, patch_args=patch_args)
    cols = set(preds_df.columns)
    assert f"{feature[NAME]}_predictions" in cols
    assert f"{feature[NAME]}_probabilities" in cols

    backend = ludwig_model.backend
    backend.df_engine.to_parquet(preds_df, os.path.join(tmpdir, "preds.parquet"))  # test saving


def predict_with_backend(tmpdir, config, data_csv_path, backend, patch_args=None):
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

    if patch_args is not None:
        with mock.patch(*patch_args):
            preds_df, _ = ludwig_model.predict(dataset=data_csv_path)
    else:
        preds_df, _ = ludwig_model.predict(dataset=data_csv_path)

    return preds_df, ludwig_model
