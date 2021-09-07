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
import logging
import os.path
import uuid
import tempfile
from unittest.mock import patch

import pytest

import mlflow
import ray
from ray.tune.sync_client import get_sync_client
from mlflow.tracking import MlflowClient

from ludwig.contribs import MlflowCallback
from ludwig.hyperopt.execution import (
    RayTuneExecutor, _get_relative_checkpoints_dir_parts, get_build_hyperopt_executor)
from ludwig.hyperopt.results import RayTuneResults
from ludwig.hyperopt.run import hyperopt
from ludwig.hyperopt.sampling import (get_build_hyperopt_sampler)
from ludwig.hyperopt.run import update_hyperopt_params_with_defaults
from ludwig.utils.defaults import merge_with_defaults, ACCURACY
from tests.integration_tests.utils import create_data_set_to_use
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import spawn
from tests.integration_tests.utils import text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

# Ray mocks

MOCK_REMOTE_DIR = os.path.join(ray._private.utils.get_user_temp_dir(),
                               "mock-tune-remote") + os.sep
# Sync and delete templates that operate on local directories.
LOCAL_SYNC_TEMPLATE = "mkdir -p {target} && rsync -avz {source}/ {target}/"
LOCAL_DELETE_TEMPLATE = "rm -rf {target}"

logger = logging.getLogger(__name__)


def mock_storage_client():
    """Mocks storage client that treats a local dir as durable storage."""
    client = get_sync_client(LOCAL_SYNC_TEMPLATE, LOCAL_DELETE_TEMPLATE)
    path = os.path.join(ray._private.utils.get_user_temp_dir(),
                        f"mock-client-{uuid.uuid4().hex[:4]}") + os.sep
    os.makedirs(path, exist_ok=True)
    client.set_logdir(path)
    return client


HYPEROPT_CONFIG = {
    "parameters": {
        "training.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.num_fc_layers": {
            "space": "randint",
            "lower": 2,
            "upper": 6
        },
        "utterance.cell_type": {
            "space": "grid_search",
            "values": ["rnn", "gru"]
        },
        "utterance.bidirectional": {
            "space": "choice",
            "categories": [True, False]
        },
        "utterance.fc_layers": {
            "space": "choice",
            "categories": [
                [{"fc_size": 512}, {"fc_size": 256}],
                [{"fc_size": 512}],
                [{"fc_size": 256}],
            ]
        }
    },
    "goal": "minimize"
}


SAMPLERS = [
    {"type": "ray"},
    {"type": "ray", "num_samples": 2},
    {
        "type": "ray",
        "search_alg": {
            "type": "bohb"
        },
        "scheduler": {
            "type": "hb_bohb",
            "time_attr": "training_iteration",
            "reduction_factor": 4,
        },
        "num_samples": 3
    },
]

EXECUTORS = [
    {"type": "ray"},
]


def _get_config(sampler, executor):
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum")]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    return {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 2, "learning_rate": 0.001},
        "hyperopt": {
            **HYPEROPT_CONFIG,
            "executor": executor,
            "sampler": sampler,
        },
    }


class MockRayTuneExecutor(RayTuneExecutor):
    def _get_sync_client_and_remote_checkpoint_dir(self, trial_dir):
        remote_checkpoint_dir = os.path.join(
            MOCK_REMOTE_DIR, *_get_relative_checkpoints_dir_parts(trial_dir))
        return mock_storage_client(), remote_checkpoint_dir

    def _validate_remote_fs_for_ray_backend(self, backend, dataset, output_directory):
        return


@pytest.fixture
def ray_start_8_cpus():
    address_info = ray.init(num_cpus=8)
    try:
        yield address_info
    finally:
        ray.shutdown()


@spawn
def run_hyperopt_executor(
    sampler, executor, csv_filename,
    validate_output_feature=False,
    validation_metric=None,
):
    config = _get_config(sampler, executor)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, 'dataset.csv')
        dataset_csv = generate_data(
            config['input_features'], config['output_features'], csv_filename)
        dataset_parquet = create_data_set_to_use('parquet', dataset_csv)

        config = merge_with_defaults(config)

        hyperopt_config = config["hyperopt"]

        if validate_output_feature:
            hyperopt_config['output_feature'] = config['output_features'][0]['name']
        if validation_metric:
            hyperopt_config['validation_metric'] = validation_metric

        update_hyperopt_params_with_defaults(hyperopt_config)

        parameters = hyperopt_config["parameters"]
        if sampler.get("search_alg", {}).get("type", "") == 'bohb':
            # bohb does not support grid_search search space
            del parameters['utterance.cell_type']

        split = hyperopt_config["split"]
        output_feature = hyperopt_config["output_feature"]
        metric = hyperopt_config["metric"]
        goal = hyperopt_config["goal"]

        hyperopt_sampler = get_build_hyperopt_sampler(
            sampler["type"])(goal, parameters, **sampler)

        hyperopt_executor = MockRayTuneExecutor(
            hyperopt_sampler, output_feature, metric, split, **executor)

        hyperopt_executor.execute(
            config,
            dataset=dataset_parquet,
            backend='ray',
            output_directory=MOCK_REMOTE_DIR
        )


@pytest.mark.distributed
@pytest.mark.parametrize('sampler', SAMPLERS)
@pytest.mark.parametrize('executor', EXECUTORS)
def test_hyperopt_executor(sampler, executor, csv_filename, ray_start_8_cpus):
    run_hyperopt_executor(sampler, executor, csv_filename)


@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename, ray_start_8_cpus):
    run_hyperopt_executor({"type": "ray", "num_samples": 2},
                          {"type": "ray"},
                          csv_filename,
                          validate_output_feature=True,
                          validation_metric=ACCURACY)


@pytest.mark.distributed
@patch("ludwig.hyperopt.execution.RayTuneExecutor", MockRayTuneExecutor)
def test_hyperopt_run_hyperopt(csv_filename, ray_start_8_cpus):
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum")]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, 'dataset.csv')
        dataset_csv = generate_data(
            input_features, output_features, csv_filename)
        dataset_parquet = create_data_set_to_use('parquet', dataset_csv)

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "num_fc_layers": 2},
            "training": {"epochs": 2, "learning_rate": 0.001}
        }

        output_feature_name = output_features[0]['name']

        hyperopt_configs = {
            "parameters": {
                "training.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
                output_feature_name + ".fc_size": {
                    "space": "randint",
                    "lower": 32,
                    "upper": 256
                },
                output_feature_name + ".num_fc_layers": {
                    "space": "randint",
                    "lower": 2,
                    "upper": 6
                }
            },
            "goal": "minimize",
            'output_feature': output_feature_name,
            'validation_metrics': 'loss',
            'executor': {'type': 'ray'},
            'sampler': {'type': 'ray', 'num_samples': 2}
        }

        # add hyperopt parameter space to the config
        config['hyperopt'] = hyperopt_configs
        run_hyperopt(config, dataset_parquet)


@pytest.mark.distributed
def test_hyperopt_ray_mlflow(csv_filename, ray_start_8_cpus, tmpdir):
    mlflow_uri = f'file://{tmpdir}/mlruns'
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)

    num_samples = 2
    config = _get_config(
        {"type": "ray", "num_samples": num_samples},
        {"type": "ray"}
    )

    rel_path = generate_data(
        config['input_features'],
        config['output_features'],
        csv_filename
    )

    exp_name = 'mlflow_test'
    run_hyperopt(config, rel_path,
                 experiment_name=exp_name,
                 callbacks=[MlflowCallback(mlflow_uri)])

    experiment = client.get_experiment_by_name(exp_name)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) > 0

    for run in runs:
        artifacts = [f.path for f in client.list_artifacts(
            run.info.run_id, "")]
        assert 'config.yaml' in artifacts
        assert 'model' in artifacts


@spawn
def run_hyperopt(
        config, rel_path,
        experiment_name='ray_hyperopt',
        callbacks=None,
):
    hyperopt_results = hyperopt(
        config,
        dataset=rel_path,
        output_directory=MOCK_REMOTE_DIR,
        experiment_name=experiment_name,
        callbacks=callbacks,
    )

    # check for return results
    assert isinstance(hyperopt_results, RayTuneResults)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(
        os.path.join('results_hyperopt', 'hyperopt_statistics.json')
    )
