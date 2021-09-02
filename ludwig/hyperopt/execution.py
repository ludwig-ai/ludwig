import datetime
import os
import uuid
import copy
import json
import multiprocessing
import signal
import shutil
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from typing import Union, Optional

from ray.tune.utils.placement_groups import PlacementGroupFactory

from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.constants import *
from ludwig.hyperopt.results import TrialResults, HyperoptResults, RayTuneResults
from ludwig.hyperopt.sampling import HyperoptSampler, RayTuneSampler, logger
from ludwig.hyperopt.utils import load_json_values
from ludwig.modules.metric_modules import get_best_function
from ludwig.utils.data_utils import NumpyEncoder
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc_utils import (get_available_gpu_memory,
                                     get_from_registry,
                                     hash_dict)
from ludwig.utils.tf_utils import get_available_gpus_cuda_string

try:
    import ray
    from ray import tune
    from ray.tune.suggest import BasicVariantGenerator, ConcurrencyLimiter
    from ray.tune.utils import wait_for_gpu
    from ray.tune import register_trainable
    from ray.util.queue import Queue as RayQueue
    from ludwig.backend.ray import RayBackend, RayRemoteTrainer, get_horovod_kwargs
except ImportError:
    ray = None
    get_horovod_kwargs = None

    class RayBackend:
        pass

    class RayRemoteTrainer:
        pass


class HyperoptExecutor(ABC):
    def __init__(self, hyperopt_sampler: Union[dict, HyperoptSampler],
                 output_feature: str, metric: str, split: str) -> None:
        self.hyperopt_sampler = hyperopt_sampler
        self.output_feature = output_feature
        self.metric = metric
        self.split = split

    def _has_metric(self, stats, split):
        if not stats:
            return False

        if split is not None:
            if split not in stats:
                return False
            stats = stats[split]

        if self.output_feature not in stats:
            return False
        stats = stats[self.output_feature]

        if self.metric not in stats:
            return False
        stats = stats[self.metric]
        return len(stats) > 0

    def _has_eval_metric(self, stats):
        if stats is None:
            return False

        if self.output_feature not in stats:
            return False
        stats = stats[self.output_feature]

        for metric_part in self.metric.split('.'):
            if not isinstance(stats, dict) or metric_part not in stats:
                return False
            stats = stats[metric_part]
        return isinstance(stats, float)

    def get_metric_score(self, train_stats, eval_stats) -> float:
        if self._has_metric(train_stats, TEST):
            logger.info(
                "Returning metric score from training (test) statistics")
            return self.get_metric_score_from_train_stats(train_stats, TEST)
        elif self._has_eval_metric(eval_stats):
            logger.info("Returning metric score from eval statistics. "
                        "If skip_save_model is True, eval statistics "
                        "are calculated using the model at the last epoch "
                        "rather than the model at the epoch with "
                        "best validation performance")
            return self.get_metric_score_from_eval_stats(eval_stats)
        elif self._has_metric(train_stats, VALIDATION):
            logger.info(
                "Returning metric score from training (validation) statistics")
            return self.get_metric_score_from_train_stats(train_stats, VALIDATION)
        elif self._has_metric(train_stats, TRAINING):
            logger.info("Returning metric score from training split statistics, "
                        "as no test / validation / eval sets were given")
            return self.get_metric_score_from_train_stats(train_stats, TRAINING)
        else:
            raise RuntimeError(
                "Unable to obtain metric score from missing training / eval statistics")

    def get_metric_score_from_eval_stats(self, eval_stats) -> Union[float, list]:
        stats = eval_stats[self.output_feature]
        for metric_part in self.metric.split('.'):
            if isinstance(stats, dict):
                if metric_part in stats:
                    stats = stats[metric_part]
                else:
                    raise ValueError(
                        f"Evaluation statistics do not contain "
                        f"the metric {self.metric}")
            else:
                raise ValueError(f"Evaluation statistics do not contain "
                                 f"the metric {self.metric}")

        if not isinstance(stats, float):
            raise ValueError(f"The metric {self.metric} in "
                             f"evaluation statistics is not "
                             f"a numerical value: {stats}")
        return stats

    def get_metric_score_from_train_stats(self, train_stats, select_split=None, returned_split=None) -> float:
        select_split = select_split or VALIDATION
        returned_split = returned_split or self.split
        if not self._has_metric(train_stats, returned_split):
            returned_split = select_split

        # grab the results of the model with highest validation test performance
        train_valiset_stats = train_stats[select_split]
        train_evalset_stats = train_stats[returned_split]

        validation_field_result = train_valiset_stats[self.output_feature]
        best_function = get_best_function(self.metric)

        # results of the model with highest validation test performance
        epoch_best_vali_metric, best_vali_metric = best_function(
            enumerate(validation_field_result[self.metric]),
            key=lambda pair: pair[1]
        )
        best_vali_metric_epoch_eval_metric = train_evalset_stats[
            self.output_feature][self.metric][
            epoch_best_vali_metric]

        return best_vali_metric_epoch_eval_metric

    def sort_hyperopt_results(self, hyperopt_results):
        return sorted(
            hyperopt_results, key=lambda hp_res: hp_res.metric_score,
            reverse=self.hyperopt_sampler.goal == MAXIMIZE
        )

    @abstractmethod
    def execute(
            self,
            config,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name="hyperopt",
            model_name="run",
            model_load_path=None,
            model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=True,
            skip_save_unprocessed_output=False,
            skip_save_predictions=False,
            skip_save_eval_stats=False,
            output_directory="results",
            gpus=None,
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            callbacks=None,
            backend=None,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ) -> HyperoptResults:
        pass


class SerialExecutor(HyperoptExecutor):
    def __init__(
            self, hyperopt_sampler: HyperoptSampler,
            output_feature: str,
            metric: str, split: str, **kwargs
    ) -> None:
        HyperoptExecutor.__init__(self, hyperopt_sampler, output_feature,
                                  metric, split)

    def execute(
            self,
            config,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=True,
            skip_save_unprocessed_output=False,
            skip_save_predictions=False,
            skip_save_eval_stats=False,
            output_directory="results",
            gpus=None,
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            callbacks=None,
            backend=None,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ) -> HyperoptResults:
        trial_results = []
        trials = 0
        while not self.hyperopt_sampler.finished():
            sampled_parameters = self.hyperopt_sampler.sample_batch()
            metric_scores = []

            for i, parameters in enumerate(sampled_parameters):
                modified_config = substitute_parameters(
                    copy.deepcopy(config), parameters)

                trial_id = trials + i

                model = LudwigModel(
                    config=modified_config,
                    backend=backend,
                    gpus=gpus,
                    gpu_memory_limit=gpu_memory_limit,
                    allow_parallel_threads=allow_parallel_threads,
                    callbacks=callbacks,
                )
                eval_stats, train_stats, _, _ = model.experiment(
                    dataset=dataset,
                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    training_set_metadata=training_set_metadata,
                    data_format=data_format,
                    experiment_name=f'{experiment_name}_{trial_id}',
                    model_name=model_name,
                    # model_load_path=model_load_path,
                    # model_resume_path=model_resume_path,
                    eval_split=self.split,
                    skip_save_training_description=skip_save_training_description,
                    skip_save_training_statistics=skip_save_training_statistics,
                    skip_save_model=skip_save_model,
                    skip_save_progress=skip_save_progress,
                    skip_save_log=skip_save_log,
                    skip_save_processed_input=skip_save_processed_input,
                    skip_save_unprocessed_output=skip_save_unprocessed_output,
                    skip_save_predictions=skip_save_predictions,
                    skip_save_eval_stats=skip_save_eval_stats,
                    output_directory=output_directory,
                    skip_collect_predictions=True,
                    skip_collect_overall_stats=False,
                    random_seed=random_seed,
                    debug=debug,
                )
                metric_score = self.get_metric_score(train_stats, eval_stats)
                metric_scores.append(metric_score)

                trial_results.append(TrialResults(
                    parameters=parameters,
                    metric_score=metric_score,
                    training_stats=train_stats,
                    eval_stats=eval_stats,
                ))
            trials += len(sampled_parameters)

            self.hyperopt_sampler.update_batch(
                zip(sampled_parameters, metric_scores))

        ordered_trials = self.sort_hyperopt_results(trial_results)
        return HyperoptResults(ordered_trials=ordered_trials)


class ParallelExecutor(HyperoptExecutor):
    num_workers = 2
    epsilon = 0.01
    epsilon_memory = 100
    TF_REQUIRED_MEMORY_PER_WORKER = 100

    def __init__(
            self,
            hyperopt_sampler: HyperoptSampler,
            output_feature: str,
            metric: str,
            split: str,
            num_workers: int = 2,
            epsilon: float = 0.01,
            **kwargs
    ) -> None:
        HyperoptExecutor.__init__(self, hyperopt_sampler, output_feature,
                                  metric, split)
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.queue = None

    @staticmethod
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _run_experiment(self, hyperopt_dict: dict) -> TrialResults:
        parameters = hyperopt_dict["parameters"]
        train_stats, eval_stats = run_experiment(**hyperopt_dict)
        metric_score = self.get_metric_score(train_stats, eval_stats)

        return TrialResults(
            parameters=parameters,
            metric_score=metric_score,
            training_stats=train_stats,
            eval_stats=eval_stats,
        )

    def _run_experiment_gpu(self, hyperopt_dict: dict) -> TrialResults:
        gpu_id_meta = self.queue.get()
        try:
            parameters = hyperopt_dict['parameters']
            hyperopt_dict["gpus"] = gpu_id_meta["gpu_id"]
            hyperopt_dict["gpu_memory_limit"] = gpu_id_meta["gpu_memory_limit"]
            train_stats, eval_stats = run_experiment(**hyperopt_dict)
            metric_score = self.get_metric_score(train_stats, eval_stats)
        finally:
            self.queue.put(gpu_id_meta)
        return TrialResults(
            parameters=parameters,
            metric_score=metric_score,
            training_stats=train_stats,
            eval_stats=eval_stats,
        )

    def execute(
            self,
            config,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=True,
            skip_save_unprocessed_output=False,
            skip_save_predictions=False,
            skip_save_eval_stats=False,
            output_directory="results",
            gpus=None,
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            callbacks=None,
            backend=None,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ) -> HyperoptResults:
        ctx = multiprocessing.get_context('spawn')

        if gpus is None:
            gpus = get_available_gpus_cuda_string()

        if gpus is not None:

            num_available_cpus = ctx.cpu_count()

            if self.num_workers > num_available_cpus:
                logger.warning(
                    "WARNING: num_workers={}, num_available_cpus={}. "
                    "To avoid bottlenecks setting num workers to be less "
                    "or equal to number of available cpus is suggested".format(
                        self.num_workers, num_available_cpus
                    )
                )

            if isinstance(gpus, int):
                gpus = str(gpus)
            gpus = gpus.strip()
            gpu_ids = gpus.split(",")
            num_gpus = len(gpu_ids)

            available_gpu_memory_list = get_available_gpu_memory()
            gpu_ids_meta = {}

            if num_gpus < self.num_workers:
                fraction = (num_gpus / self.num_workers) - self.epsilon
                for gpu_id in gpu_ids:
                    available_gpu_memory = available_gpu_memory_list[
                        int(gpu_id)]
                    required_gpu_memory = fraction * available_gpu_memory

                    if gpu_memory_limit is None:
                        logger.warning(
                            'WARNING: Setting gpu_memory_limit to {} '
                            'as there available gpus are {} '
                            'and the num of workers is {} '
                            'and the available gpu memory for gpu_id '
                            '{} is {}'.format(
                                required_gpu_memory, num_gpus,
                                self.num_workers,
                                gpu_id, available_gpu_memory)
                        )
                        new_gpu_memory_limit = required_gpu_memory - \
                            (
                                self.TF_REQUIRED_MEMORY_PER_WORKER * self.num_workers)
                    else:
                        new_gpu_memory_limit = gpu_memory_limit
                        if new_gpu_memory_limit > available_gpu_memory:
                            logger.warning(
                                'WARNING: Setting gpu_memory_limit to available gpu '
                                'memory {} minus an epsilon as the value specified is greater than '
                                'available gpu memory.'.format(
                                    available_gpu_memory)
                            )
                            new_gpu_memory_limit = available_gpu_memory - self.epsilon_memory

                        if required_gpu_memory < new_gpu_memory_limit:
                            if required_gpu_memory > 0.5 * available_gpu_memory:
                                if available_gpu_memory != new_gpu_memory_limit:
                                    logger.warning(
                                        'WARNING: Setting gpu_memory_limit to available gpu '
                                        'memory {} minus an epsilon as the gpus would be underutilized for '
                                        'the parallel processes otherwise'.format(
                                            available_gpu_memory)
                                    )
                                    new_gpu_memory_limit = available_gpu_memory - self.epsilon_memory
                            else:
                                logger.warning(
                                    'WARNING: Setting gpu_memory_limit to {} '
                                    'as the available gpus are {} and the num of workers '
                                    'are {} and the available gpu memory for gpu_id '
                                    '{} is {}'.format(
                                        required_gpu_memory, num_gpus,
                                        self.num_workers,
                                        gpu_id, available_gpu_memory)
                                )
                                new_gpu_memory_limit = required_gpu_memory
                        else:
                            logger.warning(
                                'WARNING: gpu_memory_limit could be increased to {} '
                                'as the available gpus are {} and the num of workers '
                                'are {} and the available gpu memory for gpu_id '
                                '{} is {}'.format(
                                    required_gpu_memory, num_gpus,
                                    self.num_workers,
                                    gpu_id, available_gpu_memory)
                            )

                    process_per_gpu = int(
                        available_gpu_memory / new_gpu_memory_limit)
                    gpu_ids_meta[gpu_id] = {
                        "gpu_memory_limit": new_gpu_memory_limit,
                        "process_per_gpu": process_per_gpu}
            else:
                for gpu_id in gpu_ids:
                    gpu_ids_meta[gpu_id] = {
                        "gpu_memory_limit": gpu_memory_limit,
                        "process_per_gpu": 1}

            manager = ctx.Manager()
            self.queue = manager.Queue()

            for gpu_id in gpu_ids:
                process_per_gpu = gpu_ids_meta[gpu_id]["process_per_gpu"]
                gpu_memory_limit = gpu_ids_meta[gpu_id]["gpu_memory_limit"]
                for _ in range(process_per_gpu):
                    gpu_id_meta = {"gpu_id": gpu_id,
                                   "gpu_memory_limit": gpu_memory_limit}
                    self.queue.put(gpu_id_meta)

        pool = ctx.Pool(self.num_workers,
                        ParallelExecutor.init_worker)
        try:
            trial_results = []
            trials = 0
            while not self.hyperopt_sampler.finished():
                sampled_parameters = self.hyperopt_sampler.sample_batch()

                hyperopt_parameters = []
                for i, parameters in enumerate(sampled_parameters):
                    modified_config = substitute_parameters(
                        copy.deepcopy(config), parameters)

                    trial_id = trials + i
                    hyperopt_parameters.append(
                        dict(
                            parameters=parameters,
                            config=modified_config,
                            eval_split=self.split,
                            dataset=dataset,
                            training_set=training_set,
                            validation_set=validation_set,
                            test_set=test_set,
                            training_set_metadata=training_set_metadata,
                            data_format=data_format,
                            experiment_name=f'{experiment_name}_{trial_id}',
                            model_name=model_name,
                            # model_load_path=model_load_path,
                            # model_resume_path=model_resume_path,
                            skip_save_training_description=skip_save_training_description,
                            skip_save_training_statistics=skip_save_training_statistics,
                            skip_save_model=skip_save_model,
                            skip_save_progress=skip_save_progress,
                            skip_save_log=skip_save_log,
                            # needed because of concurrent HDF5 writes
                            skip_save_processed_input=True,
                            skip_save_unprocessed_output=skip_save_unprocessed_output,
                            skip_save_predictions=skip_save_predictions,
                            skip_save_eval_stats=skip_save_eval_stats,
                            output_directory=output_directory,
                            gpus=gpus,
                            gpu_memory_limit=gpu_memory_limit,
                            allow_parallel_threads=allow_parallel_threads,
                            callbacks=callbacks,
                            backend=backend,
                            random_seed=random_seed,
                            debug=debug,
                        )
                    )
                trials += len(sampled_parameters)

                if gpus is not None:
                    batch_results = pool.map(self._run_experiment_gpu,
                                             hyperopt_parameters)
                else:
                    batch_results = pool.map(self._run_experiment,
                                             hyperopt_parameters)

                self.hyperopt_sampler.update_batch(
                    (result.parameters, result.metric_score)
                    for result in batch_results
                )

                trial_results.extend(batch_results)
        finally:
            pool.close()
            pool.join()

        ordered_trials = self.sort_hyperopt_results(trial_results)
        return HyperoptResults(ordered_trials=ordered_trials)


class FiberExecutor(HyperoptExecutor):
    num_workers = 2
    fiber_backend = "local"

    def __init__(
            self,
            hyperopt_sampler: HyperoptSampler,
            output_feature: str,
            metric: str,
            split: str,
            num_workers: int = 2,
            num_cpus_per_worker: int = -1,
            num_gpus_per_worker: int = -1,
            fiber_backend: str = "local",
            **kwargs
    ) -> None:
        import fiber

        HyperoptExecutor.__init__(self, hyperopt_sampler, output_feature,
                                  metric, split)

        fiber.init(backend=fiber_backend)
        self.fiber_meta = fiber.meta

        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker

        self.resource_limits = {}
        if num_cpus_per_worker != -1:
            self.resource_limits["cpu"] = num_cpus_per_worker

        if num_gpus_per_worker != -1:
            self.resource_limits["gpu"] = num_gpus_per_worker

        self.num_workers = num_workers
        self.pool = fiber.Pool(num_workers)

    def execute(
            self,
            config,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=True,
            skip_save_unprocessed_output=False,
            skip_save_predictions=False,
            skip_save_eval_stats=False,
            output_directory="results",
            gpus=None,
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            callbacks=None,
            backend=None,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ) -> HyperoptResults:
        experiment_kwargs = dict(
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            model_name=model_name,
            # model_load_path=model_load_path,
            # model_resume_path=model_resume_path,
            eval_split=self.split,
            skip_save_training_description=skip_save_training_description,
            skip_save_training_statistics=skip_save_training_statistics,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            skip_save_processed_input=skip_save_processed_input,
            skip_save_unprocessed_output=skip_save_unprocessed_output,
            skip_save_predictions=skip_save_predictions,
            skip_save_eval_stats=skip_save_eval_stats,
            output_directory=output_directory,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            callbacks=callbacks,
            backend=backend,
            random_seed=random_seed,
            debug=debug,
        )

        experiemnt_fn = _run_experiment_unary
        if self.resource_limits:
            experiemnt_fn = self.fiber_meta(**self.resource_limits)(
                experiemnt_fn)

        trial_results = []
        trials = 0
        while not self.hyperopt_sampler.finished():
            sampled_parameters = self.hyperopt_sampler.sample_batch()
            metric_scores = []

            stats_batch = self.pool.map(
                experiemnt_fn,
                [
                    {
                        'config': substitute_parameters(
                            copy.deepcopy(config), parameters),
                        'parameters': parameters,
                        'experiment_name': f'{experiment_name}_{trials + i}',
                        **experiment_kwargs
                    }
                    for i, parameters in enumerate(sampled_parameters)
                ],
            )
            trials += len(sampled_parameters)

            for stats, parameters in zip(stats_batch, sampled_parameters):
                train_stats, eval_stats = stats
                metric_score = self.get_metric_score(train_stats, eval_stats)
                metric_scores.append(metric_score)

                trial_results.append(TrialResults(
                    parameters=parameters,
                    metric_score=metric_score,
                    training_stats=train_stats,
                    eval_stats=eval_stats,
                ))

            self.hyperopt_sampler.update_batch(
                zip(sampled_parameters, metric_scores))

        ordered_trials = self.sort_hyperopt_results(trial_results)
        return HyperoptResults(ordered_trials=ordered_trials)


class RayTuneExecutor(HyperoptExecutor):
    def __init__(
            self,
            hyperopt_sampler,
            output_feature: str,
            metric: str,
            split: str,
            cpu_resources_per_trial: int = None,
            gpu_resources_per_trial: int = None,
            kubernetes_namespace: str = None,
            time_budget_s: Union[int, float, datetime.timedelta] = None,
            max_concurrent_trials: Optional[int] = None,
            **kwargs
    ) -> None:
        if ray is None:
            raise ImportError('ray module is not installed. To '
                              'install it,try running pip install ray'
                              )
        if not isinstance(hyperopt_sampler, RayTuneSampler):
            raise ValueError('Sampler {} is not compatible with RayTuneExecutor, '
                             'please use the RayTuneSampler'.format(
                                 hyperopt_sampler)
                             )
        HyperoptExecutor.__init__(self, hyperopt_sampler, output_feature,
                                  metric, split)
        if not ray.is_initialized():
            try:
                ray.init('auto', ignore_reinit_error=True)
            except ConnectionError:
                logger.info('Initializing new Ray cluster...')
                ray.init(ignore_reinit_error=True)
        self.search_space = hyperopt_sampler.search_space
        self.num_samples = hyperopt_sampler.num_samples
        self.goal = hyperopt_sampler.goal
        self.search_alg_dict = hyperopt_sampler.search_alg_dict
        self.scheduler = hyperopt_sampler.scheduler
        self.decode_ctx = hyperopt_sampler.decode_ctx
        self.output_feature = output_feature
        self.metric = metric
        self.split = split
        self.trial_id = 0
        self.cpu_resources_per_trial = cpu_resources_per_trial
        self.gpu_resources_per_trial = gpu_resources_per_trial
        self.kubernetes_namespace = kubernetes_namespace
        self.time_budget_s = time_budget_s
        self.max_concurrent_trials = max_concurrent_trials

    def _run_experiment(self, config, checkpoint_dir, hyperopt_dict, decode_ctx, is_using_ray_backend=False):
        for gpu_id in ray.get_gpu_ids():
            # Previous trial may not have freed its memory yet, so wait to avoid OOM
            wait_for_gpu(gpu_id)
        # Some config values may be JSON encoded as strings, so decode them here
        config = RayTuneSampler.decode_values(config, decode_ctx)

        trial_id = tune.get_trial_id()
        modified_config = substitute_parameters(
            copy.deepcopy(hyperopt_dict["config"]), config
        )

        hyperopt_dict['config'] = modified_config
        hyperopt_dict['experiment_name '] = f'{hyperopt_dict["experiment_name"]}_{trial_id}'

        tune_executor = self
        if is_using_ray_backend:
            ray_queue = RayQueue(actor_options={"num_cpus": 0})
        else:
            ray_queue = None

        def checkpoint(progress_tracker, save_path):
            with tune.checkpoint_dir(step=progress_tracker.epoch) as checkpoint_dir:
                checkpoint_model = os.path.join(checkpoint_dir, 'model')
                # shutil.copytree(save_path, checkpoint_model)
                # Note: A previous implementation used shutil.copytree()
                # however, this copying method is non atomic
                if not os.path.isdir(checkpoint_model):
                    copy_id = uuid.uuid4()
                    tmp_dst = "%s.%s.tmp" % (checkpoint_model, copy_id)
                    shutil.copytree(save_path, tmp_dst)
                    try:
                        os.rename(tmp_dst, checkpoint_model)
                    except:
                        shutil.rmtree(tmp_dst)

        def report(progress_tracker):
            train_stats = {
                TRAINING: progress_tracker.train_metrics,
                VALIDATION: progress_tracker.vali_metrics,
                TEST: progress_tracker.test_metrics,
            }

            metric_score = tune_executor.get_metric_score(
                train_stats, eval_stats=None)
            tune.report(
                parameters=json.dumps(config, cls=NumpyEncoder),
                metric_score=metric_score,
                training_stats=json.dumps(
                    train_stats[TRAINING], cls=NumpyEncoder),
                eval_stats=json.dumps(
                    train_stats[VALIDATION], cls=NumpyEncoder),
                trial_id=tune.get_trial_id(),
                trial_dir=tune.get_trial_dir()
            )

        class RayTuneReportCallback(Callback):
            def on_epoch_end(self, trainer, progress_tracker, save_path):

                if not trainer.is_coordinator():
                    return

                if isinstance(trainer, RayRemoteTrainer):
                    ray_queue.put((progress_tracker, save_path))
                    return

                checkpoint(progress_tracker, save_path)
                report(progress_tracker)

        callbacks = hyperopt_dict.get('callbacks') or []
        hyperopt_dict['callbacks'] = callbacks + [RayTuneReportCallback()]

        # set tune resources
        if is_using_ray_backend:
            resources = tune.get_trial_resources()
            # check if we are using at least 1 gpu per trial
            use_gpu = bool(self.gpu_resources_per_trial)
            # get the resources assigned to the current trial
            current_resources = resources.required_resources["GPU" if use_gpu else "CPU"]
            if not use_gpu:
                current_resources -= 1

            hyperopt_dict['backend']._horovod_kwargs['num_slots'] = None
            hyperopt_dict['backend']._horovod_kwargs['num_hosts'] = None
            hyperopt_dict['backend']._horovod_kwargs['num_workers'] = int(current_resources)
            hyperopt_dict['backend']._horovod_kwargs['use_gpu'] = use_gpu

            logger.debug(
                f"Trial horovod kwargs: {hyperopt_dict['backend']._horovod_kwargs}")

        stats = []

        def _run():
            train_stats, eval_stats = run_experiment(
                **hyperopt_dict,
                model_resume_path=checkpoint_dir,
                parameters=config,
            )
            stats.append((train_stats, eval_stats))

        if is_using_ray_backend:
            # We have to pull the results to the trial actor
            # from worker actors, as the Tune session is running
            # only on the trial actor
            thread = threading.Thread(target=_run)
            thread.daemon = True
            thread.start()
            while thread.is_alive():
                thread.join(timeout=0)
                qsize = ray_queue.qsize()
                if qsize:
                    results = ray_queue.get_nowait_batch(qsize)
                    for progress_tracker, save_path in results:
                        checkpoint(progress_tracker, save_path)
                        report(progress_tracker)
                time.sleep(0.1)
            thread.join()
        else:
            # remove threading overhead
            _run()

        if not stats:
            raise RuntimeError("Experiment did not complete.")
        train_stats, eval_stats = stats.pop()

        metric_score = self.get_metric_score(train_stats, eval_stats)
        tune.report(
            parameters=json.dumps(config, cls=NumpyEncoder),
            metric_score=metric_score,
            training_stats=json.dumps(train_stats, cls=NumpyEncoder),
            eval_stats=json.dumps(eval_stats, cls=NumpyEncoder),
            trial_id=tune.get_trial_id(),
            trial_dir=tune.get_trial_dir()
        )

    def execute(
            self,
            config,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name="hyperopt",
            model_name="run",
            # model_load_path=None,
            # model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=True,
            skip_save_unprocessed_output=False,
            skip_save_predictions=False,
            skip_save_eval_stats=False,
            output_directory="results",
            gpus=None,
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            callbacks=None,
            backend=None,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ) -> RayTuneResults:
        if isinstance(dataset, str) and "://" not in dataset and not os.path.isabs(dataset):
            dataset = os.path.abspath(dataset)

        if gpus is not None:
            raise ValueError("Parameter `gpus` is not supported when using Ray Tune. "
                             "Configure GPU resources with Ray and set `gpu_resources_per_trial` in your "
                             "hyperopt config.")

        hyperopt_dict = dict(
            config=config,
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            experiment_name=experiment_name,
            model_name=model_name,
            # model_load_path=model_load_path,
            # model_resume_path=model_resume_path,
            eval_split=self.split,
            skip_save_training_description=skip_save_training_description,
            skip_save_training_statistics=skip_save_training_statistics,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            skip_save_processed_input=skip_save_processed_input,
            skip_save_unprocessed_output=skip_save_unprocessed_output,
            skip_save_predictions=skip_save_predictions,
            skip_save_eval_stats=skip_save_eval_stats,
            output_directory=output_directory,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            callbacks=callbacks,
            backend=backend,
            random_seed=random_seed,
            debug=debug,
        )

        mode = "min" if self.goal != MAXIMIZE else "max"
        metric = "metric_score"
        if self.search_alg_dict is not None:
            if TYPE not in self.search_alg_dict:
                logger.warning(
                    "WARNING: Kindly set type param for search_alg "
                    "to utilize Tune's Search Algorithms."
                )
                search_alg = None
            else:
                search_alg_type = self.search_alg_dict.pop(TYPE)
                search_alg = tune.create_searcher(
                    search_alg_type, metric=metric, mode=mode, **self.search_alg_dict)
        else:
            search_alg = None

        if self.max_concurrent_trials:
            assert self.max_concurrent_trials > 0, f"`max_concurrent_trials` must be greater than 0, got {self.max_concurrent_trials}"
            if isinstance(search_alg, BasicVariantGenerator) or search_alg is None:
                search_alg = BasicVariantGenerator(
                    max_concurrent=self.max_concurrent_trials)
            elif isinstance(search_alg, ConcurrencyLimiter):
                raise ValueError(
                    "You have specified `max_concurrent_trials`, but the search "
                    "algorithm is already a `ConcurrencyLimiter`. FIX THIS "
                    "by setting `max_concurrent_trials=None`."
                )
            else:
                search_alg = ConcurrencyLimiter(
                    search_alg, max_concurrent=self.max_concurrent_trials)

        resources_per_trial = {
            "cpu": self.cpu_resources_per_trial or 1,
            "gpu": self.gpu_resources_per_trial or 0,
        }

        def run_experiment_trial(config, checkpoint_dir=None):
            return self._run_experiment(config, checkpoint_dir, hyperopt_dict, self.decode_ctx, isinstance(backend, RayBackend))

        tune_config = {}
        tune_callbacks = []
        for callback in callbacks or []:
            run_experiment_trial, tune_config = callback.prepare_ray_tune(
                run_experiment_trial,
                tune_config,
                tune_callbacks,
            )

        if isinstance(backend, RayBackend):
            resources_per_trial = PlacementGroupFactory(
                [{"CPU": 1}] + ([{"CPU": 1, "GPU": 1}] * self.gpu_resources_per_trial) if self.gpu_resources_per_trial else ([{"CPU": 1}] * self.cpu_resources_per_trial)
            )

        sync_config = None
        if "://" in output_directory:
            run_experiment_trial = tune.durable(run_experiment_trial)
            sync_config = tune.SyncConfig(
                sync_to_driver=False,
                upload_dir=output_directory
            )
            output_directory = None
        elif self.kubernetes_namespace:
            from ray.tune.integration.kubernetes import NamespacedKubernetesSyncer
            sync_config = tune.SyncConfig(
                sync_to_driver=NamespacedKubernetesSyncer(
                    self.kubernetes_namespace)
            )

        register_trainable(
            f"trainable_func_f{hash_dict(config).decode('ascii')}",
            run_experiment_trial
        )

        analysis = tune.run(
            f"trainable_func_f{hash_dict(config).decode('ascii')}",
            config={
                **self.search_space,
                **tune_config,
            },
            scheduler=self.scheduler,
            search_alg=search_alg,
            num_samples=self.num_samples,
            keep_checkpoints_num=1,
            resources_per_trial=resources_per_trial,
            time_budget_s=self.time_budget_s,
            queue_trials=False,
            sync_config=sync_config,
            local_dir=output_directory,
            metric=metric,
            mode=mode,
            trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            callbacks=tune_callbacks,
        )

        ordered_trials = analysis.results_df.sort_values(
            "metric_score",
            ascending=self.goal != MAXIMIZE
        )

        # Catch nans in edge case where the trial doesn't complete
        temp_ordered_trials = []
        for kwargs in ordered_trials.to_dict(orient="records"):
            for key in ['parameters', 'training_stats', 'eval_stats']:
                if isinstance(kwargs[key], float):
                    kwargs[key] = {}
            temp_ordered_trials.append(kwargs)

        ordered_trials = [
            TrialResults.from_dict(
                load_json_values(kwargs)
            )
            for kwargs in temp_ordered_trials
        ]

        return RayTuneResults(
            ordered_trials=ordered_trials,
            experiment_analysis=analysis
        )


def get_build_hyperopt_executor(executor_type):
    return get_from_registry(executor_type, executor_registry)


executor_registry = {
    "serial": SerialExecutor,
    "parallel": ParallelExecutor,
    "fiber": FiberExecutor,
    "ray": RayTuneExecutor
}


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


def substitute_parameters(config, parameters):
    parameters_dict = get_parameters_dict(parameters)
    for input_feature in config["input_features"]:
        set_values(input_feature, input_feature[COLUMN], parameters_dict)
    for output_feature in config["output_features"]:
        set_values(output_feature, output_feature[COLUMN], parameters_dict)
    set_values(config["combiner"], "combiner", parameters_dict)
    set_values(config["training"], "training", parameters_dict)
    set_values(config["preprocessing"], "preprocessing",
               parameters_dict)
    return config


def run_experiment(
        config,
        parameters=None,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        data_format=None,
        experiment_name="hyperopt",
        model_name="run",
        # model_load_path=None,
        model_resume_path=None,
        eval_split=VALIDATION,
        skip_save_training_description=False,
        skip_save_training_statistics=False,
        skip_save_model=False,
        skip_save_progress=False,
        skip_save_log=False,
        skip_save_processed_input=False,
        skip_save_unprocessed_output=False,
        skip_save_predictions=False,
        skip_save_eval_stats=False,
        output_directory="results",
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        callbacks=None,
        backend=None,
        random_seed=default_random_seed,
        debug=False,
        **kwargs
):
    for callback in callbacks or []:
        callback.on_hyperopt_trial_start(parameters)

    # Collect training and validation losses and metrics
    # & append it to `results`
    model = LudwigModel(
        config=config,
        backend=backend,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        callbacks=callbacks,
    )
    eval_stats, train_stats, _, _ = model.experiment(
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        # model_load_path=model_load_path,
        model_resume_path=model_resume_path,
        eval_split=eval_split,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        output_directory=output_directory,
        skip_collect_predictions=True,
        skip_collect_overall_stats=False,
        random_seed=random_seed,
        debug=debug,
    )
    return train_stats, eval_stats


def _run_experiment_unary(kwargs):
    """Unary function is needed by Fiber to map a list of args."""
    return run_experiment(**kwargs)
