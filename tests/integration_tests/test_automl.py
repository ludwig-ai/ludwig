import contextlib
import os
import tempfile
from typing import Any, Dict, List, Set
from unittest import mock

import pytest
from packaging import version

from ludwig.api import LudwigModel
from ludwig.constants import INPUT_FEATURES, NAME, OUTPUT_FEATURES, TRAINER
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    minio_test_creds,
    number_feature,
    private_param,
    remote_tmpdir,
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
            assert best_model.config[TRAINER]["early_stop"] == -1
            assert mock_fn.call_count == 0
        else:
            assert best_model is None
            assert mock_fn.call_count > 0
