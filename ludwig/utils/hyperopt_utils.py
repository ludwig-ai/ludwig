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
import functools
import itertools
import logging
import math
import os
import random
import signal
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import psutil

import numpy as np

from ludwig.constants import EXECUTOR, STRATEGY, MINIMIZE, COMBINED, LOSS, VALIDATION, MAXIMIZE, TRAINING, TEST
from ludwig.data.postprocessing import postprocess
from ludwig.predict import predict, print_test_results, save_prediction_outputs, save_test_statistics
from ludwig.train import full_train
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc import get_from_registry, set_default_values, \
    set_default_value

logger = logging.getLogger(__name__)


def int_sampling_function(low, high, **kwargs):
    return random.randint(low, high)


def float_sampling_function(low, high, scale='linear', base=None, **kwargs):
    if scale == 'linear':
        sample = random.uniform(low, high)
    elif scale == 'log':
        if base:
            sample = math.pow(base, random.uniform(low, high))
        else:
            sample = math.exp(random.uniform(math.log(low), math.log(high)))
    else:
        raise ValueError(
            'The scale parameter of the float sampling function is "{}". '
            'Available ones are: {"linear", "log"}'
        )
    return sample


def category_sampling_function(values, **kwargs):
    return random.sample(values, 1)[0]


def int_grid_function(low: int, high: int, steps=None, **kwargs):
    if steps is None:
        steps = high - low + 1
    samples = np.linspace(low, high, num=steps, dtype=int)
    return samples.tolist()


def float_grid_function(low, high, steps=None, scale='linear', base=None, **kwargs):
    if steps is None:
        steps = int(high - low + 1)
    if scale == 'linear':
        samples = np.linspace(low, high, num=steps)
    elif scale == 'log':
        if base:
            samples = np.logspace(low, high, num=steps, base=base)
        else:
            samples = np.geomspace(low, high, num=steps)
    else:
        raise ValueError(
            'The scale parameter of the float grid function is "{}". '
            'Available ones are: {"linear", "log"}'
        )
    return samples.tolist()


def category_grid_function(values, **kwargs):
    return values


sampling_functions_registry = {
    'int': int_sampling_function,
    'float': float_sampling_function,
    'category': category_sampling_function,
}

grid_functions_registry = {
    'int': int_grid_function,
    'float': float_grid_function,
    'category': category_grid_function
}


class HyperoptStrategy(ABC):
    def __init__(self, goal: str, parameters: Dict[str, Any]) -> None:
        self.goal = goal  # useful for bayesian stratiegy
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
                # Logic: is samples is empty it means that we encoutnered
                # the IndexError the first time we called self.sample()
                # so we should raise the exception. If samples is not empty
                # we should just return it, even if it will contain
                # less samples than the specified batch_size.
                # This is fine as from now on finished() will return True.
                if not samples:
                    raise IndexError
        return samples

    @abstractmethod
    def update(
            self,
            sampled_parameters: Dict[str, Any],
            statistics: Dict[str, Any]
    ):
        # Given the results of previous computation, it updates
        # the strategy (not needed for stateless strategies like "grid"
        # and random, but will be needed by bayesian)
        pass

    @abstractmethod
    def finished(self) -> bool:
        # Should return true when all samples have been sampled
        pass


class RandomStrategy(HyperoptStrategy):
    def __init__(
            self,
            goal: str,
            parameters: Dict[str, Any],
            num_samples=10,
            **kwargs
    ) -> None:
        HyperoptStrategy.__init__(self, goal, parameters)
        self.num_samples = num_samples
        self.samples = self._determine_samples()
        self.sampled_so_far = 0

    def _determine_samples(self):
        samples = []
        for _ in range(self.num_samples):
            sample = {}
            for hp_name, hp_params in self.parameters.items():
                sampling_function = get_from_registry(
                    hp_params['type'], sampling_functions_registry
                )
                sample[hp_name] = sampling_function(**hp_params)
            samples.append(sample)
        return samples

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
        pass

    def finished(self) -> bool:
        return self.sampled_so_far >= len(self.samples)


class GridStrategy(HyperoptStrategy):
    def __init__(self, goal: str, parameters: Dict[str, Any], **kwargs) -> None:
        HyperoptStrategy.__init__(self, goal, parameters)
        self.search_space = self._create_search_space()
        self.samples = self._get_grids()
        self.sampled_so_far = 0

    def _create_search_space(self):
        search_space = {}
        for hp_name, hp_params in self.parameters.items():
            grid_function = get_from_registry(
                hp_params['type'], grid_functions_registry
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


class HyperoptExecutor(ABC):
    def __init__(
            self,
            hyperopt_strategy: HyperoptStrategy,
            output_feature: str,
            metric: str,
            split: str
    ) -> None:
        self.hyperopt_strategy = hyperopt_strategy
        self.output_feature = output_feature
        self.metric = metric
        self.split = split

    def get_metric_score(self, eval_stats):
        return eval_stats[self.output_feature][self.metric]

    def sort_hyperopt_results(self, hyperopt_results):
        return sorted(
            hyperopt_results,
            key=lambda hp_res: hp_res['metric_score'],
            reverse=self.hyperopt_strategy.goal == MAXIMIZE
        )

    @abstractmethod
    def execute(
            self,
            model_definition,
            data_df=None,
            data_train_df=None,
            data_validation_df=None,
            data_test_df=None,
            data_csv=None,
            data_train_csv=None,
            data_validation_csv=None,
            data_test_csv=None,
            data_hdf5=None,
            data_train_hdf5=None,
            data_validation_hdf5=None,
            data_test_hdf5=None,
            train_set_metadata_json=None,
            experiment_name="hyperopt",
            model_name="run",
            model_load_path=None,
            model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            skip_save_unprocessed_output=False,
            skip_save_test_predictions=False,
            skip_save_test_statistics=False,
            output_directory="results",
            gpus=None,
            gpu_fraction=1.0,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        pass


class SerialExecutor(HyperoptExecutor):
    def __init__(
            self,
            hyperopt_strategy: HyperoptStrategy,
            output_feature: str,
            measure: str,
            split: str,
            **kwargs
    ) -> None:
        HyperoptExecutor.__init__(
            self, hyperopt_strategy, output_feature, measure, split
        )

    def execute(
            self,
            model_definition,
            data_df=None,
            data_train_df=None,
            data_validation_df=None,
            data_test_df=None,
            data_csv=None,
            data_train_csv=None,
            data_validation_csv=None,
            data_test_csv=None,
            data_hdf5=None,
            data_train_hdf5=None,
            data_validation_hdf5=None,
            data_test_hdf5=None,
            train_set_metadata_json=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            skip_save_unprocessed_output=False,
            skip_save_test_predictions=False,
            skip_save_test_statistics=False,
            output_directory="results",
            gpus=None,
            gpu_fraction=1.0,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        hyperopt_results = []
        while not self.hyperopt_strategy.finished():
            sampled_parameters = self.hyperopt_strategy.sample_batch()

            for parameters in sampled_parameters:
                modified_model_definition = substitute_parameters(
                    copy.deepcopy(model_definition), parameters
                )

                train_stats, eval_stats = train_and_eval_on_split(
                    modified_model_definition,
                    eval_split=self.split,
                    data_df=data_df,
                    data_train_df=data_train_df,
                    data_validation_df=data_validation_df,
                    data_test_df=data_test_df,
                    data_csv=data_csv,
                    data_train_csv=data_train_csv,
                    data_validation_csv=data_validation_csv,
                    data_test_csv=data_test_csv,
                    data_hdf5=data_hdf5,
                    data_train_hdf5=data_train_hdf5,
                    data_validation_hdf5=data_validation_hdf5,
                    data_test_hdf5=data_test_hdf5,
                    train_set_metadata_json=train_set_metadata_json,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    # model_load_path=model_load_path,
                    # model_resume_path=model_resume_path,
                    skip_save_training_description=skip_save_training_description,
                    skip_save_training_statistics=skip_save_training_statistics,
                    skip_save_model=skip_save_model,
                    skip_save_progress=skip_save_progress,
                    skip_save_log=skip_save_log,
                    skip_save_processed_input=skip_save_processed_input,
                    skip_save_unprocessed_output=skip_save_unprocessed_output,
                    skip_save_test_predictions=skip_save_test_predictions,
                    skip_save_test_statistics=skip_save_test_statistics,
                    output_directory=output_directory,
                    gpus=gpus,
                    gpu_fraction=gpu_fraction,
                    use_horovod=use_horovod,
                    random_seed=random_seed,
                    debug=debug,
                )
                # eval_stats = random_eval_stats(self.output_feature)
                metric_score = self.get_metric_score(eval_stats)

                hyperopt_results.append({
                    'parameters': parameters,
                    'metric_score': metric_score,
                    'training_stats': train_stats,
                    'eval_stats': eval_stats
                })

        hyperopt_results = self.sort_hyperopt_results(hyperopt_results)

        return hyperopt_results


class ParallelExecutor(HyperoptExecutor):
    def __init__(
            self,
            hyperopt_strategy: HyperoptStrategy,
            output_feature: str,
            measure: str,
            split: str,
            num_workers: int = 2,
            epsilon: int = 0.01,
            **kwargs
    ) -> None:
        HyperoptExecutor.__init__(
            self, hyperopt_strategy, output_feature, measure, split
        )
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.queue = None

    @staticmethod
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _train_and_eval_model(self, hyperopt_dict):
        parameters = hyperopt_dict['parameters']
        train_stats, eval_stats = train_and_eval_on_split(**hyperopt_dict)
        metric_score = self.get_metric_score(eval_stats)

        return {
            'parameters': parameters,
            'metric_score': metric_score,
            'training_stats': train_stats,
            'eval_stats': eval_stats
        }

    def _train_and_eval_model_gpu(self, hyperopt_dict):
        gpu_id = self.queue.get()
        try:
            parameters = hyperopt_dict['parameters']
            hyperopt_dict["gpus"] = gpu_id
            train_stats, eval_stats = train_and_eval_on_split(**hyperopt_dict)
            metric_score = self.get_metric_score(eval_stats)
        finally:
            self.queue.put(gpu_id)
        return {
            'parameters': parameters,
            'metric_score': metric_score,
            'training_stats': train_stats,
            'eval_stats': eval_stats
        }

    def execute(
            self,
            model_definition,
            data_df=None,
            data_train_df=None,
            data_validation_df=None,
            data_test_df=None,
            data_csv=None,
            data_train_csv=None,
            data_validation_csv=None,
            data_test_csv=None,
            data_hdf5=None,
            data_train_hdf5=None,
            data_validation_hdf5=None,
            data_test_hdf5=None,
            train_set_metadata_json=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            skip_save_unprocessed_output=False,
            skip_save_test_predictions=False,
            skip_save_test_statistics=False,
            output_directory="results",
            gpus=None,
            gpu_fraction=1.0,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        hyperopt_parameters = []

        if gpus is not None:

            num_available_cpus = psutil.cpu_count(logical=False)

            if num_available_cpus is None:
                num_available_cpus = psutil.cpu_count()
                logger.warning(
                    'WARNING: Couldn\'t get the number of physical cores '
                    'from the OS, using the logical cores instead.'
                )

            if self.num_workers > num_available_cpus:
                logger.warning(
                    'WARNING: Setting num_workers to less '
                    'or equal to number of available cpus: {} is suggested'.format(
                        num_available_cpus)
                )

            if isinstance(gpus, int):
                gpus = str(gpus)
            gpus = gpus.strip()
            gpu_ids = gpus.split(',')
            total_gpus = len(gpu_ids)

            if total_gpus < self.num_workers:
                fraction = (total_gpus / self.num_workers) - self.epsilon
                if fraction < gpu_fraction:
                    if fraction > 0.5:
                        if gpu_fraction != 1:
                            logger.warning(
                                'WARNING: Setting gpu_fraction to 1 as the gpus '
                                'would be underutilized for the parallel processes.'
                            )
                        gpu_fraction = 1
                    else:
                        logger.warning(
                            'WARNING: Setting gpu_fraction to {} '
                            'as the available gpus is {} and the num of workers '
                            'selected is {}'.format(
                                fraction, total_gpus, self.num_workers)
                        )
                        gpu_fraction = fraction
                else:
                    logger.warning(
                        'WARNING: gpu_fraction could be increased to {} '
                        'as the available gpus is {} and the num of workers '
                        'being set is {}'.format(
                            fraction, total_gpus, self.num_workers)
                    )

            process_per_gpu = int(1 / gpu_fraction)

            manager = multiprocessing.Manager()
            self.queue = manager.Queue()

            for gpu_id in gpu_ids:
                for _ in range(process_per_gpu):
                    self.queue.put(gpu_id)

        while not self.hyperopt_strategy.finished():
            sampled_parameters = self.hyperopt_strategy.sample_batch()

            for parameters in sampled_parameters:
                modified_model_definition = substitute_parameters(
                    copy.deepcopy(model_definition), parameters
                )

                hyperopt_parameters.append(
                    {
                        'parameters': parameters,
                        'model_definition': modified_model_definition,
                        'eval_split': self.split,
                        'data_df': data_df,
                        'data_train_df': data_train_df,
                        'data_validation_df': data_validation_df,
                        'data_test_df': data_test_df,
                        'data_csv': data_csv,
                        'data_train_csv': data_train_csv,
                        'data_validation_csv': data_validation_csv,
                        'data_test_csv': data_test_csv,
                        'data_hdf5': data_hdf5,
                        'data_train_hdf5': data_train_hdf5,
                        'data_validation_hdf5': data_validation_hdf5,
                        'data_test_hdf5': data_test_hdf5,
                        'train_set_metadata_json': train_set_metadata_json,
                        'experiment_name': experiment_name,
                        'model_name': model_name,
                        # model_load_path:model_load_path,
                        # model_resume_path:model_resume_path,
                        'skip_save_training_description': skip_save_training_description,
                        'skip_save_training_statistics': skip_save_training_statistics,
                        'skip_save_model': skip_save_model,
                        'skip_save_progress': skip_save_progress,
                        'skip_save_log': skip_save_log,
                        'skip_save_processed_input': skip_save_processed_input,
                        'skip_save_unprocessed_output': skip_save_unprocessed_output,
                        'skip_save_test_predictions': skip_save_test_predictions,
                        'skip_save_test_statistics': skip_save_test_statistics,
                        'output_directory': output_directory,
                        'gpus': gpus,
                        'gpu_fraction': gpu_fraction,
                        'use_horovod': use_horovod,
                        'random_seed': random_seed,
                        'debug': debug,
                    }
                )

        pool = multiprocessing.Pool(
            self.num_workers, ParallelExecutor.init_worker)

        if gpus is not None:
            hyperopt_results = pool.map(
                self._train_and_eval_model_gpu, hyperopt_parameters)
        else:
            hyperopt_results = pool.map(
                self._train_and_eval_model, hyperopt_parameters)

        hyperopt_results = self.sort_hyperopt_results(hyperopt_results)
        return hyperopt_results


class FiberExecutor(HyperoptExecutor):
    def __init__(
            self,
            hyperopt_strategy: HyperoptStrategy,
            output_feature: str,
            measure: str,
            split: str,
            num_workers: int = 2,
            fiber_backend: str = "local",
            **kwargs
    ) -> None:
        import fiber
        HyperoptExecutor.__init__(
            self, hyperopt_strategy, output_feature, measure, split
        )
        fiber.init(backend=fiber_backend)
        self.pool = fiber.Pool(num_workers)

    def execute(
            self,
            model_definition,
            data_df=None,
            data_train_df=None,
            data_validation_df=None,
            data_test_df=None,
            data_csv=None,
            data_train_csv=None,
            data_validation_csv=None,
            data_test_csv=None,
            data_hdf5=None,
            data_train_hdf5=None,
            data_validation_hdf5=None,
            data_test_hdf5=None,
            train_set_metadata_json=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            skip_save_unprocessed_output=False,
            skip_save_test_predictions=False,
            skip_save_test_statistics=False,
            output_directory="results",
            gpus=None,
            gpu_fraction=1.0,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        train_func = functools.partial(
            train_and_eval_on_split,
            eval_split=self.split,
            data_df=data_df,
            data_train_df=data_train_df,
            data_validation_df=data_validation_df,
            data_test_df=data_test_df,
            data_csv=data_csv,
            data_train_csv=data_train_csv,
            data_validation_csv=data_validation_csv,
            data_test_csv=data_test_csv,
            data_hdf5=data_hdf5,
            data_train_hdf5=data_train_hdf5,
            data_validation_hdf5=data_validation_hdf5,
            data_test_hdf5=data_test_hdf5,
            train_set_metadata_json=train_set_metadata_json,
            experiment_name=experiment_name,
            model_name=model_name,
            # model_load_path=model_load_path,
            # model_resume_path=model_resume_path,
            skip_save_training_description=skip_save_training_description,
            skip_save_training_statistics=skip_save_training_statistics,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            skip_save_processed_input=skip_save_processed_input,
            skip_save_unprocessed_output=skip_save_unprocessed_output,
            skip_save_test_predictions=skip_save_test_predictions,
            skip_save_test_statistics=skip_save_test_statistics,
            output_directory=output_directory,
            gpus=gpus,
            gpu_fraction=gpu_fraction,
            use_horovod=use_horovod,
            random_seed=random_seed,
            debug=debug,
        )

        hyperopt_results = []
        while not self.hyperopt_strategy.finished():
            sampled_parameters = self.hyperopt_strategy.sample_batch()

            stats_batch = self.pool.map(
                train_func,
                [
                    substitute_parameters(
                        copy.deepcopy(model_definition),parameters
                    )
                    for parameters in sampled_parameters
                ]
            )

            for stats, parameters in zip(stats_batch, sampled_parameters):
                train_stats, eval_stats = stats
                metric_score = self.get_metric_score(eval_stats)

                hyperopt_results.append({
                    'parameters': parameters,
                    'metric_score': metric_score,
                    'training_stats': train_stats,
                    'eval_stats': eval_stats
                })

        hyperopt_results = self.sort_hyperopt_results(hyperopt_results)

        return hyperopt_results


def get_build_hyperopt_strategy(strategy_type):
    return get_from_registry(
        strategy_type, strategy_registry
    )


def get_build_hyperopt_executor(executor_type):
    return get_from_registry(
        executor_type, executor_registry
    )


strategy_registry = {
    "random": RandomStrategy,
    "grid": GridStrategy
}

executor_registry = {
    "serial": SerialExecutor,
    "parallel": ParallelExecutor,
    "fiber": FiberExecutor,
}


# TODO this function should read default parameters from strategies / executors
def update_hyperopt_params_with_defaults(hyperopt_params):
    set_default_value(hyperopt_params, STRATEGY, {})
    set_default_value(hyperopt_params, EXECUTOR, {})
    set_default_value(hyperopt_params, 'split', VALIDATION)
    set_default_value(hyperopt_params, 'output_feature', COMBINED)
    set_default_value(hyperopt_params, 'metric', LOSS)
    set_default_value(hyperopt_params, 'goal', MINIMIZE)

    set_default_values(
        hyperopt_params[STRATEGY],
        {
            "type": "random",
            "num_samples": 12
        }
    )

    if hyperopt_params[STRATEGY]["type"] == "grid":
        set_default_values(
            hyperopt_params[STRATEGY],
            {
                # Put Grid default values
            },
        )

    set_default_values(
        hyperopt_params[EXECUTOR],
        {
            "type": "serial"
        }
    )

    if hyperopt_params[EXECUTOR]["type"] == "parallel":
        set_default_values(
            hyperopt_params[EXECUTOR],
            {
                "num_workers": 4
            }
        )


def set_values(model_dict, name, parameters_dict):
    if name in parameters_dict:
        params = parameters_dict[name]
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    model_dict[key][sub_key] = sub_value
            else:
                model_dict[key] = value


def get_parameters_dict(parameters):
    parameters_dict = {}
    for name, value in parameters.items():
        curr_dict = parameters_dict
        name_list = name.split(".")
        for i, name_elem in enumerate(name_list):
            if i == len(name_list) - 1:
                curr_dict[name_elem] = value
            else:
                name_dict = curr_dict.get(name_elem, {})
                curr_dict[name_elem] = name_dict
                curr_dict = name_dict
    return parameters_dict


def substitute_parameters(model_definition, parameters):
    parameters_dict = get_parameters_dict(parameters)
    for input_feature in model_definition["input_features"]:
        set_values(input_feature, input_feature["name"], parameters_dict)
    for output_feature in model_definition["output_features"]:
        set_values(output_feature, output_feature["name"], parameters_dict)
    set_values(model_definition["combiner"], "combiner", parameters_dict)
    set_values(model_definition["training"], "training", parameters_dict)
    set_values(model_definition["preprocessing"],
               "preprocessing", parameters_dict)
    return model_definition


# TODo this is duplicate code from experiment,
#  reorganize experiment to avoid having to do this
def train_and_eval_on_split(
        model_definition,
        eval_split=VALIDATION,
        data_df=None,
        data_train_df=None,
        data_validation_df=None,
        data_test_df=None,
        data_csv=None,
        data_train_csv=None,
        data_validation_csv=None,
        data_test_csv=None,
        data_hdf5=None,
        data_train_hdf5=None,
        data_validation_hdf5=None,
        data_test_hdf5=None,
        train_set_metadata_json=None,
        experiment_name="hyperopt",
        model_name="run",
        # model_load_path=None,
        # model_resume_path=None,
        skip_save_training_description=False,
        skip_save_training_statistics=False,
        skip_save_model=False,
        skip_save_progress=False,
        skip_save_log=False,
        skip_save_processed_input=False,
        skip_save_unprocessed_output=False,
        skip_save_test_predictions=False,
        skip_save_test_statistics=False,
        output_directory="results",
        gpus=None,
        gpu_fraction=1.0,
        use_horovod=False,
        random_seed=default_random_seed,
        debug=False,
        **kwargs
):
    # Collect training and validation losses and measures
    # & append it to `results`
    # ludwig_model = LudwigModel(modified_model_definition)
    (
        model,
        preprocessed_data,
        experiment_dir_name,
        train_stats,
        model_definition
    ) = full_train(
        model_definition=model_definition,
        data_df=data_df,
        data_train_df=data_train_df,
        data_validation_df=data_validation_df,
        data_test_df=data_test_df,
        data_csv=data_csv,
        data_train_csv=data_train_csv,
        data_validation_csv=data_validation_csv,
        data_test_csv=data_test_csv,
        data_hdf5=data_hdf5,
        data_train_hdf5=data_train_hdf5,
        data_validation_hdf5=data_validation_hdf5,
        data_test_hdf5=data_test_hdf5,
        train_set_metadata_json=train_set_metadata_json,
        experiment_name=experiment_name,
        model_name=model_name,
        # model_load_path=model_load_path,
        # model_resume_path=model_resume_path,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        output_directory=output_directory,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug,
    )
    (training_set,
     validation_set,
     test_set,
     train_set_metadata) = preprocessed_data
    if model_definition[TRAINING]['eval_batch_size'] > 0:
        batch_size = model_definition[TRAINING]['eval_batch_size']
    else:
        batch_size = model_definition[TRAINING]['batch_size']

    eval_set = validation_set
    if eval_split == TRAINING:
        eval_set = training_set
    elif eval_split == VALIDATION:
        eval_set = validation_set
    elif eval_split == TEST:
        eval_set = test_set

    test_results = predict(
        eval_set,
        train_set_metadata,
        model,
        model_definition,
        batch_size,
        evaluate_performance=True,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        debug=debug
    )
    if not (
            skip_save_unprocessed_output and
            skip_save_test_predictions and
            skip_save_test_statistics
    ):
        if not os.path.exists(experiment_dir_name):
            os.makedirs(experiment_dir_name)

    # postprocess
    postprocessed_output = postprocess(
        test_results,
        model_definition['output_features'],
        train_set_metadata,
        experiment_dir_name,
        skip_save_unprocessed_output
    )

    print_test_results(test_results)
    if not skip_save_test_predictions:
        save_prediction_outputs(
            postprocessed_output,
            experiment_dir_name
        )
    if not skip_save_test_statistics:
        save_test_statistics(test_results, experiment_dir_name)
    return train_stats, test_results


def random_eval_stats(output_feature):
    eval_stats = {
        'training': {
            'combined': {
                'loss': random.uniform(0.1, 2.0)
            },
            output_feature: {
                'loss': random.uniform(0.1, 2.0),
                'accuracy': random.uniform(0.0, 1.0),
                'mean_squared_error': random.uniform(0.0, 1000),
            }
        },
        'validation': {
            'combined': {
                'loss': random.uniform(0.1, 2.0)
            },
            output_feature: {
                'loss': random.uniform(0.1, 2.0),
                'accuracy': random.uniform(0.0, 1.0),
                'mean_squared_error': random.uniform(0.0, 1000),
            }
        },
        'test': {
            'combined': {
                'loss': random.uniform(0.1, 2.0)
            },
            output_feature: {
                'loss': random.uniform(0.1, 2.0),
                'accuracy': random.uniform(0.0, 1.0),
                'mean_squared_error': random.uniform(0.0, 1000),
            }
        }
    }
    return eval_stats
