import contextlib
import os
import tempfile
from typing import Any, Dict, List, Set
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from packaging import version

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, ENCODER, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PREPROCESSING, SPLIT, TYPE
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    image_feature,
    minio_test_creds,
    number_feature,
    private_param,
    remote_tmpdir,
    text_feature,
)

ray = pytest.importorskip("ray")

import dask.dataframe as dd  # noqa

from ludwig.automl.automl import create_auto_config, train_with_config  # noqa
from ludwig.hyperopt.execution import RayTuneExecutor  # noqa

_ray200 = version.parse(ray.__version__) >= version.parse("2.0")

pytestmark = pytest.mark.distributed


@pytest.fixture(scope="module")
def test_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [
            number_feature(),
            number_feature(),
            category_feature(encoder={"vocab_size": 3}),
            category_feature(encoder={"vocab_size": 3}),
        ]
        output_features = [category_feature(decoder={"vocab_size": 3})]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100
        )
        yield input_features, output_features, dataset_csv


@pytest.mark.distributed
@pytest.mark.parametrize("tune_for_memory", [True, False])
def test_create_auto_config(tune_for_memory, test_data, ray_cluster_2cpu):
    input_features, output_features, dataset_csv = test_data
    targets = [feature[NAME] for feature in output_features]
    df = dd.read_csv(dataset_csv)
    config = create_auto_config(df, targets, time_limit_s=600, tune_for_memory=tune_for_memory, backend="ray")

    def to_name_set(features: List[Dict[str, Any]]) -> Set[str]:
        return {feature[NAME] for feature in features}

    assert to_name_set(config[INPUT_FEATURES]) == to_name_set(input_features)
    assert to_name_set(config[OUTPUT_FEATURES]) == to_name_set(output_features)


def _get_sample_df(class_probs):
    nrows = 1000
    thresholds = np.cumsum((class_probs * nrows).astype(int))

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])

    def get_category(v):
        if v < thresholds[0]:
            return 0
        if thresholds[0] <= v < thresholds[1]:
            return 1
        return 2

    df["category"] = df.index.map(get_category).astype(np.int8)
    return df


@pytest.mark.distributed
def test_autoconfig_preprocessing_balanced():
    df = _get_sample_df(np.array([0.33, 0.33, 0.34]))

    config = create_auto_config(dataset=df, target="category", time_limit_s=1, tune_for_memory=False)

    assert PREPROCESSING not in config


@pytest.mark.distributed
def test_autoconfig_preprocessing_imbalanced():
    df = _get_sample_df(np.array([0.6, 0.2, 0.2]))

    config = create_auto_config(dataset=df, target="category", time_limit_s=1, tune_for_memory=False)

    assert PREPROCESSING in config
    assert SPLIT in config[PREPROCESSING]
    assert config[PREPROCESSING][SPLIT] == {TYPE: "stratify", COLUMN: "category"}


@pytest.mark.distributed
def test_autoconfig_preprocessing_text_image(tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    input_features = [text_feature(preprocessing={"tokenizer": "space"}), image_feature(folder=image_dest_folder)]
    output_features = [category_feature(output_feature=True)]

    # Generate Dataset
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))
    df = pd.read_csv(rel_path)
    target = df.columns[-1]

    config = create_auto_config(dataset=df, target=target, time_limit_s=1, tune_for_memory=False)

    # Check no features shuffled around
    assert len(input_features) == 2
    assert len(output_features) == 1

    # Check encoders are properly nested
    assert isinstance(config[INPUT_FEATURES][0][ENCODER], dict)
    assert isinstance(config[INPUT_FEATURES][1][ENCODER], dict)

    # Check automl default encoders are properly set
    assert config[INPUT_FEATURES][0][ENCODER][TYPE] == "bert"
    assert config[INPUT_FEATURES][1][ENCODER][TYPE] == "stacked_cnn"


@pytest.mark.distributed
@pytest.mark.parametrize("time_budget", [200, 1], ids=["high", "low"])
def test_train_with_config(time_budget, test_data, ray_cluster_2cpu, tmpdir):
    _run_train_with_config(time_budget, test_data, tmpdir)


@pytest.mark.parametrize("fs_protocol,bucket", [private_param(("s3", "ludwig-tests"))], ids=["s3"])
def test_train_with_config_remote(fs_protocol, bucket, test_data, ray_cluster_2cpu):
    backend = {
        "type": "local",
        "credentials": {
            "artifacts": minio_test_creds(),
        },
    }

    with remote_tmpdir(fs_protocol, bucket) as tmpdir:
        with pytest.raises(ValueError) if not _ray200 else contextlib.nullcontext():
            _run_train_with_config(200, test_data, tmpdir, backend=backend)


def _run_train_with_config(time_budget, test_data, tmpdir, **kwargs):
    input_features, output_features, dataset_csv = test_data
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {"epochs": 2},
        "hyperopt": {
            "search_alg": {
                "type": "hyperopt",
                "random_state_seed": 42,
            },
            "executor": {
                "type": "ray",
                "time_budget_s": time_budget,
                "cpu_resources_per_trial": 1,
                "scheduler": {
                    "type": "async_hyperband",
                    "max_t": time_budget,
                    "time_attr": "time_total_s",
                    "grace_period": min(72, time_budget),
                    "reduction_factor": 5,
                },
            },
            "parameters": {
                "trainer.batch_size": {
                    "space": "choice",
                    "categories": [64, 128, 256],
                },
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
            },
        },
    }

    fn = RayTuneExecutor._evaluate_best_model
    with mock.patch("ludwig.hyperopt.execution.RayTuneExecutor._evaluate_best_model") as mock_fn:
        # We need to check that _evaluate_best_model is called when the time_budget is low
        # as this code path should be triggered when the trial was early stopped
        mock_fn.side_effect = fn

        outdir = os.path.join(tmpdir, "output")
        results = train_with_config(dataset_csv, config, output_directory=outdir, **kwargs)
        best_model = results.best_model

        if time_budget > 1:
            assert isinstance(best_model, LudwigModel)
            assert best_model.config_obj.trainer.early_stop == -1
            assert mock_fn.call_count == 0
        else:
            assert best_model is None
            assert mock_fn.call_count > 0
