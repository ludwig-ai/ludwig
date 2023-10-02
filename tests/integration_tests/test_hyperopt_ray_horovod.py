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
import os.path
import shutil
import uuid
from unittest.mock import patch

import pytest

from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.constants import ACCURACY, AUTO, BATCH_SIZE, EXECUTOR, MAX_CONCURRENT_TRIALS, TRAINER
from ludwig.globals import HYPEROPT_STATISTICS_FILE_NAME
from ludwig.hyperopt.results import HyperoptResults
from ludwig.hyperopt.run import hyperopt
from ludwig.hyperopt.utils import update_hyperopt_params_with_defaults
from ludwig.schema.model_config import ModelConfig
from tests.integration_tests.utils import binary_feature, create_data_set_to_use, generate_data, number_feature

try:
    import ray
    from ray.tune.syncer import get_node_to_storage_syncer, SyncConfig

    from ludwig.backend.ray import RayBackend
    from ludwig.hyperopt.execution import _get_relative_checkpoints_dir_parts, RayTuneExecutor
except ImportError:
    ray = None
    RayTuneExecutor = object


pytestmark = pytest.mark.integration_tests_a


# Dummy sync templates
LOCAL_SYNC_TEMPLATE = "echo {source}/ {target}/"
LOCAL_DELETE_TEMPLATE = "echo {target}"


def mock_storage_client(path):
    """Mocks storage client that treats a local dir as durable storage."""
    os.makedirs(path, exist_ok=True)
    syncer = get_node_to_storage_syncer(SyncConfig(upload_dir=path))
    return syncer


HYPEROPT_CONFIG = {
    "parameters": {
        "trainer.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.output_size": {"space": "grid_search", "values": [4, 8]},
    },
    "goal": "minimize",
}


SCENARIOS = [
    {
        "executor": {
            "type": "ray",
            "num_samples": 2,
            "trial_driver_resources": {"hyperopt_resources": 1},  # Used to prevent deadlock
            "cpu_resources_per_trial": 1,
        },
        "search_alg": {"type": "variant_generator"},
    },
    {
        "executor": {
            "type": "ray",
            "num_samples": 2,
            "scheduler": {
                "type": "hb_bohb",
                "time_attr": "training_iteration",
                "reduction_factor": 4,
            },
            "trial_driver_resources": {"hyperopt_resources": 1},  # Used to prevent deadlock
            "cpu_resources_per_trial": 1,
        },
        "search_alg": {"type": "bohb"},
    },
    # TODO(shreya): Uncomment when https://github.com/ludwig-ai/ludwig/issues/2039 is fixed.
    # {
    #     "type": "ray",
    #     "num_samples": 1,
    #     "scheduler": {
    #         "type": "async_hyperband",
    #         "time_attr": "training_iteration",
    #         "reduction_factor": 2,
    #         "dynamic_resource_allocation": True,
    #     },
    # },
]


# NOTE(geoffrey): As of PR #2079, we reduce the test's processor parallelism from 4 to 1.
#
# We reduce parallelism to ensure that Ray Datasets doesn't reserve all available CPUs ahead of the other trials
# being scheduled. Before this change, all CPUs for the train_fn of each trial were scheduled up front by
# the Tuner, which meant that Ray Datasets could safely grab all remaining CPUs.
#
# In this change, only the dummy hyperopt_resources are scheduled by the Tuner. The inner Tuners then
# schedule CPUs ad-hoc as they are called and executed by each trial. The danger with this is in its interaction with
# Ray Datasets, which grabs resources opportunistically. If an inner Tuner is scheduled and its Ray Datasets tasks grab
# the remaining CPUs, other trials may be prevented from starting, causing the test to double in duration
# (since some trials are executed in sequence instead of all at once).
#
# Setting parallelism to 1 here ensures that the number of CPUs requested by Ray Datasets is limited to 1 per trial.
# For more context, see https://github.com/ludwig-ai/ludwig/pull/2709/files#r1042812690
RAY_BACKEND_KWARGS = {"processor": {"parallelism": 1}}


def _get_config(search_alg, executor):
    input_features = [number_feature()]
    output_features = [binary_feature()]

    # When using the hb_bohb scheduler, num_epochs must equal max_t (which is 81 by default)
    num_epochs = 1 if search_alg["type"] == "variant_generator" else 81

    return {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat"},
        TRAINER: {"epochs": num_epochs, "learning_rate": 0.001, BATCH_SIZE: 128},
        "hyperopt": {
            **HYPEROPT_CONFIG,
            "executor": executor,
            "search_alg": search_alg,
        },
    }


class MockRayTuneExecutor(RayTuneExecutor):
    def _get_sync_client_and_remote_checkpoint_dir(self, trial_dir):
        remote_checkpoint_dir = os.path.join(self.mock_path, *_get_relative_checkpoints_dir_parts(trial_dir))
        return mock_storage_client(remote_checkpoint_dir), remote_checkpoint_dir


class CustomTestCallback(Callback):
    def __init__(self):
        self.preprocessed = False

    def on_hyperopt_preprocessing_start(self, *args, **kwargs):
        self.preprocessed = True

    def on_hyperopt_start(self, *args, **kwargs):
        assert self.preprocessed


@pytest.fixture
def ray_mock_dir():
    path = os.path.join(ray._private.utils.get_user_temp_dir(), f"mock-client-{uuid.uuid4().hex[:4]}") + os.sep
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def run_hyperopt_executor(
    search_alg,
    executor,
    csv_filename,
    ray_mock_dir,
    validate_output_feature=False,
    validation_metric=None,
):
    config = _get_config(search_alg, executor)

    csv_filename = os.path.join(ray_mock_dir, "dataset.csv")
    dataset_csv = generate_data(config["input_features"], config["output_features"], csv_filename, num_examples=25)
    dataset_parquet = create_data_set_to_use("parquet", dataset_csv)

    config = ModelConfig.from_dict(config).to_dict()

    hyperopt_config = config["hyperopt"]

    if validate_output_feature:
        hyperopt_config["output_feature"] = config["output_features"][0]["name"]
    if validation_metric:
        hyperopt_config["validation_metric"] = validation_metric

    backend = RayBackend(**RAY_BACKEND_KWARGS)
    update_hyperopt_params_with_defaults(hyperopt_config)
    if hyperopt_config[EXECUTOR].get(MAX_CONCURRENT_TRIALS) == AUTO:
        hyperopt_config[EXECUTOR][MAX_CONCURRENT_TRIALS] = backend.max_concurrent_trials(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    if search_alg.get("type", "") == "bohb":
        # bohb does not support grid_search search space
        del parameters["combiner.output_size"]
        hyperopt_config["parameters"] = parameters

    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]
    search_alg = hyperopt_config["search_alg"]

    # preprocess
    model = LudwigModel(config=config, backend=backend)
    training_set, validation_set, test_set, training_set_metadata = model.preprocess(
        dataset=dataset_parquet,
    )

    # hyperopt
    hyperopt_executor = MockRayTuneExecutor(
        parameters, output_feature, metric, goal, split, search_alg=search_alg, **hyperopt_config[EXECUTOR]
    )
    hyperopt_executor.mock_path = os.path.join(ray_mock_dir, "bucket")

    hyperopt_executor.execute(
        config,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        backend=backend,
        output_directory=ray_mock_dir,
        skip_save_processed_input=True,
        skip_save_unprocessed_output=True,
        resume=False,
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_hyperopt_executor_variant_generator(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    search_alg = SCENARIOS[0]["search_alg"]
    executor = SCENARIOS[0]["executor"]
    run_hyperopt_executor(search_alg, executor, csv_filename, ray_mock_dir)


@pytest.mark.skip(reason="PG/resource cleanup bugs in Ray 2.x: https://github.com/ray-project/ray/issues/31738")
@pytest.mark.distributed
def test_hyperopt_executor_bohb(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    search_alg = SCENARIOS[1]["search_alg"]
    executor = SCENARIOS[1]["executor"]
    run_hyperopt_executor(search_alg, executor, csv_filename, ray_mock_dir)


@pytest.mark.distributed
@pytest.mark.skip(reason="https://github.com/ludwig-ai/ludwig/issues/1441")
@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    run_hyperopt_executor(
        {"type": "variant_generator"},  # search_alg
        {"type": "ray", "num_samples": 2},  # executor
        csv_filename,
        ray_mock_dir,
        validate_output_feature=True,
        validation_metric=ACCURACY,
    )


@pytest.mark.skip(reason="https://github.com/ludwig-ai/ludwig/issues/1441")
@pytest.mark.distributed
@patch("ludwig.hyperopt.execution.RayTuneExecutor", MockRayTuneExecutor)
def test_hyperopt_run_hyperopt(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    input_features = [number_feature()]
    output_features = [binary_feature()]

    csv_filename = os.path.join(ray_mock_dir, "dataset.csv")
    dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=100)
    dataset_parquet = create_data_set_to_use("parquet", dataset_csv)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat"},
        TRAINER: {"epochs": 1, "learning_rate": 0.001, BATCH_SIZE: 128},
        "backend": {"type": "ray", **RAY_BACKEND_KWARGS},
    }

    output_feature_name = output_features[0]["name"]

    hyperopt_configs = {
        "parameters": {
            "trainer.learning_rate": {
                "space": "loguniform",
                "lower": 0.001,
                "upper": 0.1,
            },
            output_feature_name + ".output_size": {"space": "randint", "lower": 2, "upper": 8},
        },
        "goal": "minimize",
        "output_feature": output_feature_name,
        "validation_metrics": "loss",
        "executor": {"type": "ray", "num_samples": 2},
        "search_alg": {"type": "variant_generator"},
    }

    # add hyperopt parameter space to the config
    config["hyperopt"] = hyperopt_configs
    run_hyperopt(config, dataset_parquet, ray_mock_dir)


def run_hyperopt(
    config,
    rel_path,
    out_dir,
    experiment_name="ray_hyperopt",
):
    callback = CustomTestCallback()
    hyperopt_results = hyperopt(
        config,
        dataset=rel_path,
        output_directory=out_dir,
        experiment_name=experiment_name,
        callbacks=[callback],
    )

    # check for return results
    assert isinstance(hyperopt_results, HyperoptResults)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(os.path.join(out_dir, experiment_name, HYPEROPT_STATISTICS_FILE_NAME))
