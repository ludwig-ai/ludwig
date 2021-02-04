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
import pytest

try:
    from ray import tune
except ImportError:
    RAY_AVAILABLE = False
else:
    RAY_AVAILABLE = True

from ludwig.hyperopt.sampling import RayTuneSampler


HYPEROPT_PARAMS = {
    "test_1": {
        "parameters": {
            "training.learning_rate": {
                "space": "uniform",
                "lower": 0.001,
                "upper": 0.1
            },
            "combiner.num_fc_layers": {
                "space": "qrandint",
                "lower": 3,
                "upper": 6,
                "q": 3
            },
            "utterance.cell_type": {
                "space": "grid_search",
                "values": ["rnn", "gru", "lstm"]
            }
        },
        "goal": "minimize",
        "num_samples": 3
    },
    "test_2": {
        "parameters": {
            "training.learning_rate": {
                "space": "loguniform",
                "lower": 0.001,
                "upper": 0.1,
                "base": 10,
            },
            "combiner.num_fc_layers": {
                "space": "randint",
                "lower": 2,
                "upper": 6
            },
            "utterance.cell_type": {
                "space": "choice",
                "categories": ["rnn", "gru", "lstm"]
            }
        },
        "goal": "maximize",
        "num_samples": 4
    }
}

if RAY_AVAILABLE:
    EXPECTED_SEARCH_SPACE = {
        "test_1": {
            "training.learning_rate": tune.uniform(0.001, 0.1),
            "combiner.num_fc_layers": tune.qrandint(3, 6, 3),
            "utterance.cell_type": tune.grid_search(["rnn", "gru", "lstm"])
        },
        "test_2": {
            "training.learning_rate": tune.loguniform(0.001, 0.1),
            "combiner.num_fc_layers": tune.randint(2, 6),
            "utterance.cell_type": tune.choice(["rnn", "gru", "lstm"])
        }
    }


@pytest.mark.skipif(not RAY_AVAILABLE,
                    reason="Ray is not installed for testing")
@pytest.mark.parametrize("key", ["test_1", "test_2"])
def test_grid_strategy(key):

    hyperopt_test_params = HYPEROPT_PARAMS[key]
    expected_search_space = EXPECTED_SEARCH_SPACE[key]

    goal = hyperopt_test_params["goal"]
    num_samples = hyperopt_test_params["num_samples"]
    tune_sampler_params = hyperopt_test_params["parameters"]

    tune_sampler = RayTuneSampler(goal=goal, parameters=tune_sampler_params,
                                  num_samples=num_samples)
    search_space = tune_sampler.search_space

    actual_params_keys = search_space.keys()
    expected_params_keys = expected_search_space.keys()

    for param in search_space:
        assert type(search_space[param]) is type(expected_search_space[param])

    assert actual_params_keys == expected_params_keys
    assert tune_sampler.num_samples == num_samples
