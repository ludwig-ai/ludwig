#! /usr/bin/env python
# coding=utf-8
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
import copy
import inspect
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from ray import tune

from bayesmark.builtin_opt.pysot_optimizer import PySOTOptimizer
from bayesmark.space import JointSpace
from ludwig.constants import (CATEGORY, FLOAT, INT, MAXIMIZE, MINIMIZE, SPACE,
                              TYPE)
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.strings_utils import str2bool

logger = logging.getLogger(__name__)


def int_grid_function(low: int, high: int, steps=None, **kwargs):
    if steps is None:
        steps = high - low + 1
    samples = np.linspace(low, high, num=steps, dtype=int)
    return samples.tolist()


def float_grid_function(low: float, high: float, steps=None, space='linear',
                        base=None, **kwargs):
    if steps is None:
        steps = int(high - low + 1)
    if space == 'linear':
        samples = np.linspace(low, high, num=steps)
    elif space == 'log':
        if base:
            samples = np.logspace(low, high, num=steps, base=base)
        else:
            samples = np.geomspace(low, high, num=steps)
    else:
        raise ValueError(
            'The space parameter of the float grid function is "{}". '
            'Available ones are: {"linear", "log"}'
        )
    return samples.tolist()


def category_grid_function(values, **kwargs):
    return values


grid_functions_registry = {
    'int': int_grid_function,
    'float': float_grid_function,
    'category': category_grid_function,
}


class HyperoptSampler(ABC):
    def __init__(self, goal: str, parameters: Dict[str, Any]) -> None:
        assert goal in [MINIMIZE, MAXIMIZE]
        self.goal = goal  # useful for Bayesian strategy
        self.parameters = parameters

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        # Yields a set of parameters names and their values.
        # Define `build_hyperopt_strategy` which would take paramters as inputs
        pass

    def sample_batch(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        samples = []
        for _ in range(batch_size):
            try:
                samples.append(self.sample())
            except IndexError:
                # Logic: is samples is empty it means that we encountered
                # the IndexError the first time we called self.sample()
                # so we should raise the exception. If samples is not empty
                # we should just return it, even if it will contain
                # less samples than the specified batch_size.
                # This is fine as from now on finished() will return True.
                if not samples:
                    raise IndexError
        return samples

    @abstractmethod
    def update(self, sampled_parameters: Dict[str, Any], metric_score: float):
        # Given the results of previous computation, it updates
        # the strategy (not needed for stateless strategies like "grid"
        # and random, but will be needed by Bayesian)
        pass

    def update_batch(self, parameters_metric_tuples: Iterable[
        Tuple[Dict[str, Any], float]]):
        for (sampled_parameters, metric_score) in parameters_metric_tuples:
            self.update(sampled_parameters, metric_score)

    @abstractmethod
    def finished(self) -> bool:
        # Should return true when all samples have been sampled
        pass


class RandomSampler(HyperoptSampler):
    num_samples = 10

    def __init__(self, goal: str, parameters: Dict[str, Any], num_samples=10,
                 **kwargs) -> None:
        HyperoptSampler.__init__(self, goal, parameters)
        params_for_join_space = copy.deepcopy(parameters)

        cat_params_values_types = {}
        for param_name, param_values in params_for_join_space.items():
            if param_values[TYPE] == CATEGORY:
                param_values[TYPE] = 'cat'
                values_str = []
                values_types = {}
                for value in param_values['values']:
                    value_str = str(value)
                    values_str.append(value_str)
                    value_type = type(value)
                    if value_type == bool:
                        value_type = str2bool
                    values_types[value_str] = value_type
                param_values['values'] = values_str
                cat_params_values_types[param_name] = values_types
            if param_values[TYPE] == FLOAT:
                param_values[TYPE] = 'real'
            if param_values[TYPE] == INT or param_values[TYPE] == 'real':
                if SPACE not in param_values:
                    param_values[SPACE] = 'linear'
                param_values['range'] = (param_values['low'],
                                         param_values['high'])
                del param_values['low']
                del param_values['high']

        self.cat_params_values_types = cat_params_values_types
        self.space = JointSpace(params_for_join_space)
        self.num_samples = num_samples
        self.samples = self._determine_samples()
        self.sampled_so_far = 0

    def _determine_samples(self):
        samples = []
        for _ in range(self.num_samples):
            bnds = self.space.get_bounds()
            x = bnds[:, 0] + (bnds[:, 1] - bnds[:, 0]) * np.random.rand(1, len(
                self.space.get_bounds()))
            sample = self.space.unwarp(x)[0]
            samples.append(sample)
        return samples

    def sample(self) -> Dict[str, Any]:
        if self.sampled_so_far >= len(self.samples):
            raise IndexError()
        sample = self.samples[self.sampled_so_far]
        for key in sample:
            if key in self.cat_params_values_types:
                values_types = self.cat_params_values_types[key]
                sample[key] = values_types[sample[key]](sample[key])
        self.sampled_so_far += 1
        return sample

    def update(self, sampled_parameters: Dict[str, Any], metric_score: float):
        pass

    def finished(self) -> bool:
        return self.sampled_so_far >= len(self.samples)


class GridSampler(HyperoptSampler):
    def __init__(self, goal: str, parameters: Dict[str, Any],
                 **kwargs) -> None:
        HyperoptSampler.__init__(self, goal, parameters)
        self.search_space = self._create_search_space()
        self.samples = self._get_grids()
        self.sampled_so_far = 0

    def _create_search_space(self):
        search_space = {}
        for hp_name, hp_params in self.parameters.items():
            grid_function = get_from_registry(
                hp_params[TYPE], grid_functions_registry
            )
            search_space[hp_name] = grid_function(**hp_params)
        return search_space

    def _get_grids(self):
        hp_params = sorted(self.search_space)
        grids = [dict(zip(hp_params, prod)) for prod in itertools.product(
            *(self.search_space[hp_name] for hp_name in hp_params))]

        return grids

    def sample(self) -> Dict[str, Any]:
        if self.sampled_so_far >= len(self.samples):
            raise IndexError()
        sample = self.samples[self.sampled_so_far]
        self.sampled_so_far += 1
        return sample

    def update(
            self,
            sampled_parameters: Dict[str, Any],
            statistics: Dict[str, Any]
    ):
        # actual implementation ...
        pass

    def finished(self) -> bool:
        return self.sampled_so_far >= len(self.samples)


class PySOTSampler(HyperoptSampler):
    """pySOT: Surrogate optimization in Python.
    This is a wrapper around the pySOT package (https://github.com/dme65/pySOT):
        David Eriksson, David Bindel, Christine Shoemaker
        pySOT and POAP: An event-driven asynchronous framework for surrogate optimization
    """

    def __init__(self, goal: str, parameters: Dict[str, Any], num_samples=10,
                 **kwargs) -> None:
        HyperoptSampler.__init__(self, goal, parameters)
        params_for_join_space = copy.deepcopy(parameters)

        cat_params_values_types = {}
        for param_name, param_values in params_for_join_space.items():
            if param_values[TYPE] == CATEGORY:
                param_values[TYPE] = 'cat'
                values_str = []
                values_types = {}
                for value in param_values['values']:
                    value_str = str(value)
                    values_str.append(value_str)
                    value_type = type(value)
                    if value_type == bool:
                        value_type = str2bool
                    values_types[value_str] = value_type
                param_values['values'] = values_str
                cat_params_values_types[param_name] = values_types
            if param_values[TYPE] == FLOAT:
                param_values[TYPE] = 'real'
            if param_values[TYPE] == INT or param_values[TYPE] == 'real':
                if SPACE not in param_values:
                    param_values[SPACE] = 'linear'
                param_values['range'] = (param_values['low'],
                                         param_values['high'])
                del param_values['low']
                del param_values['high']

        self.cat_params_values_types = cat_params_values_types
        self.pysot_optimizer = PySOTOptimizer(params_for_join_space)
        self.sampled_so_far = 0
        self.num_samples = num_samples

    def sample(self) -> Dict[str, Any]:
        """Suggest one new point to be evaluated."""
        if self.sampled_so_far >= self.num_samples:
            raise IndexError()
        sample = self.pysot_optimizer.suggest(n_suggestions=1)[0]
        for key in sample:
            if key in self.cat_params_values_types:
                values_types = self.cat_params_values_types[key]
                sample[key] = values_types[sample[key]](sample[key])
        self.sampled_so_far += 1
        return sample

    def update(self, sampled_parameters: Dict[str, Any], metric_score: float):
        for key in sampled_parameters:
            if key in self.cat_params_values_types:
                sampled_parameters[key] = str(sampled_parameters[key])
        self.pysot_optimizer.observe([sampled_parameters], [metric_score])

    def finished(self) -> bool:
        return self.sampled_so_far >= self.num_samples


def get_tune_search_space(parameters, num_samples):
    sampler = {}
    config = {}
    for param, values in parameters.items():
        space = values["space"].lower()
        if hasattr(tune, space):
            space_function = getattr(tune, space)
        else:
            raise ValueError(
                "'{}' method is not supported in the Ray Tune module".format(space))
        space_input_args = {}
        space_required_args = inspect.getfullargspec(space_function).args
        for arg in space_required_args:
            if arg in values:
                space_input_args[arg] = values[arg]
            else:
                raise ValueError(
                    "Parameter '{}' not defined for {}".format(arg, param))
        config[param] = space_function(**space_input_args)
    sampler["config"] = config
    sampler["num_samples"] = num_samples
    return sampler


def get_build_hyperopt_sampler(strategy_type):
    return get_from_registry(strategy_type, sampler_registry)


sampler_registry = {
    "grid": GridSampler,
    "random": RandomSampler,
    "pysot": PySOTSampler,
}
