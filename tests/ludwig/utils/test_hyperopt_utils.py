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

HYPEROPT_PARAMS = {
    "test_1": {
        "parameters": {
            "trainer.learning_rate": {"type": "float", "low": 0.0001, "high": 0.1, "steps": 4, "space": "log"},
            "combiner.num_fc_layers": {"type": "int", "low": 1, "high": 4},
            "utterance.cell_type": {"type": "category", "values": ["rnn", "gru", "lstm"]},
        },
        "expected_search_space": {
            "trainer.learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "combiner.num_fc_layers": [1, 2, 3, 4],
            "utterance.cell_type": ["rnn", "gru", "lstm"],
        },
        "goal": "minimize",
        "expected_len_grids": 48,
        "num_samples": 10,
    },
    "test_2": {
        "parameters": {
            "trainer.learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "steps": 4, "space": "linear"},
            "combiner.num_fc_layers": {"type": "int", "low": 2, "high": 6, "steps": 3},
        },
        "expected_search_space": {
            "trainer.learning_rate": [0.001, 0.034, 0.067, 0.1],
            "combiner.num_fc_layers": [2, 4, 6],
        },
        "goal": "maximize",
        "expected_len_grids": 12,
        "num_samples": 5,
    },
}


@pytest.mark.skip("Rework for RayTune")
@pytest.mark.parametrize("key", ["test_1", "test_2"])
def test_grid_strategy(key):
    hyperopt_test_params = HYPEROPT_PARAMS[key]
    goal = hyperopt_test_params["goal"]
    grid_sampler_params = hyperopt_test_params["parameters"]

    # TODO: Rework for RayTune
    # grid_sampler = GridSampler(goal=goal, parameters=grid_sampler_params)
    #
    # actual_params_keys = grid_sampler.sample().keys()
    # expected_params_keys = grid_sampler_params.keys()
    #
    # for sample in grid_sampler.samples:
    #     for param in actual_params_keys:
    #         value = sample[param]
    #         param_type = grid_sampler_params[param]["type"]
    #         if param_type == "int" or param_type == "float":
    #             low = grid_sampler_params[param]["low"]
    #             high = grid_sampler_params[param]["high"]
    #             assert value >= low and value <= high
    #         else:
    #             assert value in set(grid_sampler_params[param]["values"])
    #
    # assert actual_params_keys == expected_params_keys
    # assert grid_sampler.search_space == hyperopt_test_params["expected_search_space"]
    # assert len(grid_sampler.samples) == hyperopt_test_params["expected_len_grids"]


@pytest.mark.skip("Rework for RayTune")
@pytest.mark.parametrize("key", ["test_1", "test_2"])
def test_random_sampler(key):
    hyperopt_test_params = HYPEROPT_PARAMS[key]
    goal = hyperopt_test_params["goal"]
    random_sampler_params = hyperopt_test_params["parameters"]
    num_samples = hyperopt_test_params["num_samples"]

    # TODO: Rework for RayTune
    # random_sampler = RandomSampler(goal=goal, parameters=random_sampler_params, num_samples=num_samples)
    #
    # actual_params_keys = random_sampler.sample().keys()
    # expected_params_keys = random_sampler_params.keys()
    #
    # for sample in random_sampler.samples:
    #     for param in actual_params_keys:
    #         value = sample[param]
    #         param_type = random_sampler_params[param]["type"]
    #         if param_type == "int" or param_type == "float":
    #             low = random_sampler_params[param]["low"]
    #             high = random_sampler_params[param]["high"]
    #             assert value >= low and value <= high
    #         else:
    #             assert value in set(random_sampler_params[param]["values"])
    #
    # assert actual_params_keys == expected_params_keys
    # assert len(random_sampler.samples) == num_samples
