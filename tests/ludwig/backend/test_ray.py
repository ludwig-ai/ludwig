import copy
from unittest.mock import patch

import pytest

# Skip these tests if Ray is not installed
ray = pytest.importorskip("ray")  # noqa

from ray.train.horovod import HorovodConfig  # noqa

from ludwig.backend import initialize_backend  # noqa
from ludwig.backend.ray import get_trainer_kwargs  # noqa
from ludwig.constants import AUTO, EXECUTOR, MAX_CONCURRENT_TRIALS, RAY  # noqa

# Mark the entire module as distributed
pytestmark = pytest.mark.distributed


@pytest.mark.parametrize(
    "trainer_config,cluster_resources,num_nodes,expected_kwargs",
    [
        # Prioritize using the GPU when available over multi-node
        (
            {},
            {"CPU": 4, "GPU": 1},
            2,
            dict(
                backend=HorovodConfig(),
                num_workers=1,
                use_gpu=True,
                resources_per_worker={
                    "CPU": 0,
                    "GPU": 1,
                },
            ),
        ),
        # Use one worker per node for CPU, chck NIC override
        (
            {"nics": [""]},
            {"CPU": 4, "GPU": 0},
            2,
            dict(
                backend=HorovodConfig(nics={""}),
                num_workers=2,
                use_gpu=False,
                resources_per_worker={
                    "CPU": 1,
                    "GPU": 0,
                },
            ),
        ),
        # Allow explicitly setting GPU usage for autoscaling clusters
        (
            {"use_gpu": True, "num_workers": 2},
            {"CPU": 4, "GPU": 0},
            1,
            dict(
                backend=HorovodConfig(),
                num_workers=2,
                use_gpu=True,
                resources_per_worker={
                    "CPU": 0,
                    "GPU": 1,
                },
            ),
        ),
        # Allow overriding resources_per_worker
        (
            {"resources_per_worker": {"CPU": 2, "GPU": 1}},
            {"CPU": 4, "GPU": 2},
            2,
            dict(
                backend=HorovodConfig(),
                num_workers=2,
                use_gpu=True,
                resources_per_worker={
                    "CPU": 2,
                    "GPU": 1,
                },
            ),
        ),
    ],
)
def test_get_trainer_kwargs(trainer_config, cluster_resources, num_nodes, expected_kwargs):
    with patch("ludwig.backend.ray.ray.cluster_resources", return_value=cluster_resources):
        with patch("ludwig.backend.ray._num_nodes", return_value=num_nodes):
            trainer_config_copy = copy.deepcopy(trainer_config)
            actual_kwargs = get_trainer_kwargs(**trainer_config_copy)

            # Function should not modify the original input
            assert trainer_config_copy == trainer_config

            actual_backend = actual_kwargs.pop("backend")
            expected_backend = expected_kwargs.pop("backend")

            assert type(actual_backend) == type(expected_backend)
            assert actual_backend.nics == expected_backend.nics
            assert actual_kwargs == expected_kwargs


@pytest.mark.distributed
@pytest.mark.parametrize(
    "hyperopt_config_old, hyperopt_config_expected",
    [
        (  # If max_concurrent_trials is none, it should not be set in the updated config
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": None},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": None},
            },
        ),
        (  # If max_concurrent_trials is auto, set it to total_trials - 2 if num_samples == num_cpus
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": "auto"},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 3},
            },
        ),
        (  # Even though num_samples is set to 4, this will actually result in 9 trials. We should correctly set
            # max_concurrent_trials to 2
            {
                "parameters": {
                    "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.01, 0.1]},
                    "combiner.num_fc_layers": {"space": "grid_search", "values": [1, 2, 3]},
                },
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": "auto"},
            },
            {
                "parameters": {
                    "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.01, 0.1]},
                    "combiner.num_fc_layers": {"space": "grid_search", "values": [1, 2, 3]},
                },
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 3},
            },
        ),
        (  # Ensure user config value (1) is respected if it is passed in
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 1},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 1},
            },
        ),
    ],
    ids=["none", "auto", "auto_with_large_num_trials", "1"],
)
def test_set_max_concurrent_trials(hyperopt_config_old, hyperopt_config_expected, ray_cluster_4cpu):
    backend = initialize_backend(RAY)
    if hyperopt_config_old[EXECUTOR].get(MAX_CONCURRENT_TRIALS) == AUTO:
        hyperopt_config_old[EXECUTOR][MAX_CONCURRENT_TRIALS] = backend.max_concurrent_trials(hyperopt_config_old)
    assert hyperopt_config_old == hyperopt_config_expected
