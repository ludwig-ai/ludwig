import copy
import os
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Skip these tests if Ray is not installed
ray = pytest.importorskip("ray")

from ray.train.torch import TorchConfig  # noqa

from ludwig.backend import initialize_backend  # noqa
from ludwig.backend.ray import _load_results, _make_picklable, get_trainer_kwargs  # noqa
from ludwig.constants import AUTO, EXECUTOR, MAX_CONCURRENT_TRIALS, RAY  # noqa
from ludwig.utils.metric_utils import TrainerMetric  # noqa

# Mark the entire module as distributed
pytestmark = [pytest.mark.distributed, pytest.mark.distributed_d]


@pytest.mark.parametrize(
    "trainer_config,cluster_resources,num_nodes,expected_kwargs",
    [
        # Prioritize using the GPU when available over multi-node
        pytest.param(
            {},
            {"CPU": 4, "GPU": 1},
            2,
            dict(
                backend=TorchConfig(),
                num_workers=1,
                use_gpu=True,
                resources_per_worker={
                    "CPU": 0,
                    "GPU": 1,
                },
            ),
            id="accelerate",
            marks=pytest.mark.distributed,
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

            assert type(actual_backend) is type(expected_backend)
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
        (  # If max_concurrent_trials is auto, set to cpus // cpus_per_trial
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": "auto"},
            },
            {
                "parameters": {"trainer.learning_rate": {"space": "choice", "values": [0.001, 0.01, 0.1]}},
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 4},
            },
        ),
        (  # Even though num_samples is set to 4, this will actually result in 9 trials.
            # With 4 CPUs and 1 CPU/trial, max_concurrent_trials = 4
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
                "executor": {"num_samples": 4, "cpu_resources_per_trial": 1, "max_concurrent_trials": 4},
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


def test_load_results_roundtrip(tmp_path):
    """Training metrics saved by train_fn load back with the restricted unpickler."""
    metrics = defaultdict(lambda: defaultdict(list))
    metrics["out"]["loss"].append(TrainerMetric(epoch=1, step=10, value=np.float32(0.5)))
    metrics["out"]["accuracy"].append(TrainerMetric(epoch=1, step=10, value=torch.tensor(0.75)))
    other_results = _make_picklable([metrics, {}, {}])

    path = os.path.join(tmp_path, "train_other.pt")
    torch.save(other_results, path)
    loaded = _load_results(path)

    loss = loaded[0]["out"]["loss"][0]
    assert isinstance(loss, TrainerMetric)
    assert loss == TrainerMetric(epoch=1, step=10, value=0.5)
    assert loaded[0]["out"]["accuracy"][0].value.item() == 0.75


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_load_results_rejects_arbitrary_code(tmp_path):
    """A tampered results file on shared checkpoint storage must not execute code on the driver."""
    path = os.path.join(tmp_path, "train_other.pt")
    torch.save([{"metric": _Exploit()}], path)
    with patch("os.system") as system:
        with pytest.raises(Exception, match="Weights only load failed"):
            _load_results(path)
    system.assert_not_called()
