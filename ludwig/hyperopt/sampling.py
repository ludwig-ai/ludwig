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
from inspect import signature
from typing import Any, Dict

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


def identity(x):
    return x


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
