#! /usr/bin/env python
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
import json
import logging
from abc import ABC
from importlib import import_module
from inspect import signature
from typing import Any, Dict

import numpy as np

from ludwig.utils.misc_utils import get_from_registry

try:
    from ray import tune
    from ray.tune.schedulers.resource_changing_scheduler import (
        evenly_distribute_cpus_gpus,
        PlacementGroupFactory,
        ResourceChangingScheduler,
    )

    _HAS_RAY_TUNE = True
except ImportError:
    evenly_distribute_cpus_gpus = None
    _HAS_RAY_TUNE = False


def ray_resource_allocation_function(
    trial_runner: "trial_runner.TrialRunner",  # noqa
    trial: "Trial",  # noqa
    result: Dict[str, Any],
    scheduler: "ResourceChangingScheduler",
):
    """Determine resources to allocate to running trials."""
    pgf = evenly_distribute_cpus_gpus(trial_runner, trial, result, scheduler)
    # restore original base trial resources

    # create bundles
    if scheduler.base_trial_resources.required_resources.get("GPU", 0):
        bundles = [{"CPU": 1, "GPU": 1}] * int(pgf.required_resources["GPU"])
    else:
        bundles = [{"CPU": 1}] * (int(pgf.required_resources["CPU"] - 0.001))
    # we can't set Trial actor's CPUs to 0 so we just go very low
    bundles = [{"CPU": 0.001}] + bundles
    pgf = PlacementGroupFactory(bundles)
    return pgf


logger = logging.getLogger(__name__)


def int_grid_function(low: int, high: int, steps=None, **kwargs):
    if steps is None:
        steps = high - low + 1
    samples = np.linspace(low, high, num=steps, dtype=int)
    return samples.tolist()


def float_grid_function(low: float, high: float, steps=None, space="linear", base=None, **kwargs):
    if steps is None:
        steps = int(high - low + 1)
    if space == "linear":
        samples = np.linspace(low, high, num=steps)
    elif space == "log":
        if base:
            samples = np.logspace(low, high, num=steps, base=base)
        else:
            samples = np.geomspace(low, high, num=steps)
    else:
        raise ValueError(
            'The space parameter of the float grid function is "{}". ' 'Available ones are: {"linear", "log"}'
        )
    return samples.tolist()


def category_grid_function(values, **kwargs):
    return values


def identity(x):
    return x


grid_functions_registry = {
    "int": int_grid_function,
    "float": float_grid_function,
    "category": category_grid_function,
}


class RayTuneSampler:
    def __init__(
        self,
        parameters: Dict[str, Any],
        **kwargs,
    ) -> None:
        self._check_ray_tune()
        self.search_space, self.decode_ctx = self._get_search_space(parameters)

    def _check_ray_tune(self):
        if not _HAS_RAY_TUNE:
            raise ValueError("Requested Ray sampler but Ray Tune is not installed. Run `pip install ray[tune]`")

    def _get_search_space(self, parameters):
        config = {}
        ctx = {}
        for param, values in parameters.items():
            # Encode list and dict types as JSON encoded strings to
            # workaround type limitations of the underlying frameworks
            values = self.encode_values(param, values, ctx)

            param_search_type = values["space"].lower()
            if hasattr(tune, param_search_type):
                param_search_space = getattr(tune, param_search_type)
            else:
                raise ValueError(f"'{param_search_type}' is not a supported Ray Tune search space")

            param_search_input_args = {}
            param_search_space_sig = signature(param_search_space)
            for arg in param_search_space_sig.parameters.values():
                if arg.name in values:
                    param_search_input_args[arg.name] = values[arg.name]
                else:
                    if arg.default is arg.empty:
                        raise ValueError(f"Parameter '{arg}' not defined for {param}")
            config[param] = param_search_space(**param_search_input_args)
        return config, ctx

    @staticmethod
    def encode_values(param, values, ctx):
        """JSON encodes any search spaces whose values are lists / dicts.

        Only applies to grid search and choice options.  See here for details:

        https://docs.ray.io/en/master/tune/api_docs/search_space.html#random-distributions-api
        """
        values = values.copy()
        for key in ["values", "categories"]:
            if key in values and not isinstance(values[key][0], (int, float)):
                values[key] = [json.dumps(v) for v in values[key]]
                ctx[param] = json.loads
        return values

    @staticmethod
    def decode_values(config, ctx):
        """Decode config values with the decode function in the context.

        Uses the identity function if no encoding is needed.
        """
        return {key: ctx.get(key, identity)(value) for key, value in config.items()}


def get_build_hyperopt_sampler(strategy_type):
    return get_from_registry(strategy_type, sampler_registry)


sampler_registry = {"ray": RayTuneSampler}


# TODO: split to separate module?
def _is_package_installed(package_name: str, search_algo_name: str) -> bool:
    try:
        import_module(package_name)
        return True
    except ImportError:
        raise ImportError(
            f"Search algorithm {search_algo_name} requires package {package_name}, however package is not installed."
            " Please refer to Ray Tune documentation for packages required for this search algorithm."
        )


class SearchAlgorithm(ABC):
    def __init__(self, search_alg_dict: Dict) -> None:
        self.search_alg_dict = search_alg_dict
        self.random_seed_attribute_name = None

    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        if self.random_seed_attribute_name not in self.search_alg_dict:
            self.search_alg_dict[self.random_seed_attribute_name] = ludwig_random_seed


class BasicVariantSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state"


class HyperoptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hyperopt", "hyperopt")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


class BOHBSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hpbandster", "bohb")
        _is_package_installed("ConfigSpace", "bohb")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"


class AxSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("sqlalchemy", "ax")
        _is_package_installed("ax", "ax")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class BayesOptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("bayes_opt", "bayesopt")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state"


class BlendsearchSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("flaml", "blendsearch")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class CFOSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("flaml", "cfo")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class DragonflySA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("dragonfly", "dragonfly")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


class HEBOSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hebo", "hebo")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


def get_search_algorithm(search_algo):
    return get_from_registry(search_algo, search_algo_registry)


search_algo_registry = {
    None: BasicVariantSA,
    "variant_generator": BasicVariantSA,
    "random": BasicVariantSA,
    "hyperopt": HyperoptSA,
    "bohb": BOHBSA,
    "ax": AxSA,
    "bayesopt": BayesOptSA,
    "blendsearch": BlendsearchSA,
    "cfo": CFOSA,
    "dragonfly": DragonflySA,
    "hebo": HEBOSA,
}
