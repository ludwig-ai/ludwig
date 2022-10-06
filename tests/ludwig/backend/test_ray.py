import copy
import os
from unittest.mock import patch

import pytest

# Skip these tests if Ray is not installed
ray = pytest.importorskip("ray")  # noqa

from ray.train.constants import TRAIN_ENABLE_WORKER_SPREAD_ENV  # noqa
from ray.train.horovod import HorovodConfig  # noqa

from ludwig.backend.ray import get_trainer_kwargs, spread_env  # noqa

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


@pytest.mark.parametrize(
    "trainer_kwargs,current_env_value,expected_env_value",
    [
        ({"use_gpu": False}, None, "1"),
        ({"use_gpu": True}, None, None),
        ({"use_gpu": True}, "1", "1"),
        ({"use_gpu": True}, "", ""),
    ],
)
def test_spread_env(trainer_kwargs, current_env_value, expected_env_value):
    prev_env = os.environ.get(TRAIN_ENABLE_WORKER_SPREAD_ENV)

    # Set environment to state prior to override
    if current_env_value is not None:
        os.environ[TRAIN_ENABLE_WORKER_SPREAD_ENV] = current_env_value
    elif TRAIN_ENABLE_WORKER_SPREAD_ENV in os.environ:
        del os.environ[TRAIN_ENABLE_WORKER_SPREAD_ENV]

    with spread_env(trainer_kwargs):
        assert os.environ.get(TRAIN_ENABLE_WORKER_SPREAD_ENV) == expected_env_value
    assert os.environ.get(TRAIN_ENABLE_WORKER_SPREAD_ENV) == current_env_value

    # Return environment to original state
    if prev_env is not None:
        os.environ[TRAIN_ENABLE_WORKER_SPREAD_ENV] = prev_env
    elif TRAIN_ENABLE_WORKER_SPREAD_ENV in os.environ:
        del os.environ[TRAIN_ENABLE_WORKER_SPREAD_ENV]
