import copy
import multiprocessing
import os
import signal
from abc import ABC, abstractmethod

from ludwig.constants import MAXIMIZE, VALIDATION, TRAINING, TEST
from ludwig.data.postprocessing import postprocess
from ludwig.hyperopt.sampling import HyperoptSamplingStrategy, \
    logger
from ludwig.predict import predict, print_test_results, \
    save_prediction_outputs, save_test_statistics
from ludwig.train import full_train
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc import get_available_gpu_memory, get_from_registry
from ludwig.utils.tf_utils import get_available_gpus_cuda_string


class HyperoptExecutor(ABC):
    def __init__(self, hyperopt_strategy: HyperoptSamplingStrategy,
                 output_feature: str, metric: str, split: str) -> None:
        self.hyperopt_strategy = hyperopt_strategy
        self.output_feature = output_feature
        self.metric = metric
        self.split = split

    def get_metric_score(self, eval_stats) -> float:
        return eval_stats[self.output_feature][self.metric]

    def sort_hyperopt_results(self, hyperopt_results):
        return sorted(
            hyperopt_results, key=lambda hp_res: hp_res["metric_score"],
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
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        pass


class SerialExecutor(HyperoptExecutor):
    def __init__(
            self, hyperopt_strategy: HyperoptSamplingStrategy,
            output_feature: str,
            metric: str, split: str, **kwargs
    ) -> None:
        HyperoptExecutor.__init__(self, hyperopt_strategy, output_feature,
                                  metric, split)

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
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        hyperopt_results = []
        trials = 0
        while not self.hyperopt_strategy.finished():
            sampled_parameters = self.hyperopt_strategy.sample_batch()
            metric_scores = []

            for i, parameters in enumerate(sampled_parameters):
                modified_model_definition = substitute_parameters(
                    copy.deepcopy(model_definition), parameters)

                trial_id = trials + i
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
                    experiment_name=f'{experiment_name}_{trial_id}',
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
                    gpu_memory_limit=gpu_memory_limit,
                    allow_parallel_threads=allow_parallel_threads,
                    use_horovod=use_horovod,
                    random_seed=random_seed,
                    debug=debug,
                )
                metric_score = self.get_metric_score(eval_stats)
                metric_scores.append(metric_score)

                hyperopt_results.append(
                    {
                        "parameters": parameters,
                        "metric_score": metric_score,
                        "training_stats": train_stats,
                        "eval_stats": eval_stats,
                    }
                )
            trials += len(sampled_parameters)

            self.hyperopt_strategy.update_batch(
                zip(sampled_parameters, metric_scores))

        hyperopt_results = self.sort_hyperopt_results(hyperopt_results)

        return hyperopt_results


class ParallelExecutor(HyperoptExecutor):
    num_workers = 2
    epsilon = 0.01
    epsilon_memory = 100
    TF_REQUIRED_MEMORY_PER_WORKER = 100

    def __init__(
            self,
            hyperopt_strategy: HyperoptSamplingStrategy,
            output_feature: str,
            metric: str,
            split: str,
            num_workers: int = 2,
            epsilon: float = 0.01,
            **kwargs
    ) -> None:
        HyperoptExecutor.__init__(self, hyperopt_strategy, output_feature,
                                  metric, split)
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.queue = None

    @staticmethod
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _train_and_eval_model(self, hyperopt_dict):
        parameters = hyperopt_dict["parameters"]
        train_stats, eval_stats = train_and_eval_on_split(**hyperopt_dict)
        metric_score = self.get_metric_score(eval_stats)

        return {
            "parameters": parameters,
            "metric_score": metric_score,
            "training_stats": train_stats,
            "eval_stats": eval_stats,
        }

    def _train_and_eval_model_gpu(self, hyperopt_dict):
        gpu_id_meta = self.queue.get()
        try:
            parameters = hyperopt_dict['parameters']
            hyperopt_dict["gpus"] = gpu_id_meta["gpu_id"]
            hyperopt_dict["gpu_memory_limit"] = gpu_id_meta["gpu_memory_limit"]
            train_stats, eval_stats = train_and_eval_on_split(**hyperopt_dict)
            metric_score = self.get_metric_score(eval_stats)
        finally:
            self.queue.put(gpu_id_meta)
        return {
            "parameters": parameters,
            "metric_score": metric_score,
            "training_stats": train_stats,
            "eval_stats": eval_stats,
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
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
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
            hyperopt_results = []
            trials = 0
            while not self.hyperopt_strategy.finished():
                sampled_parameters = self.hyperopt_strategy.sample_batch()

                hyperopt_parameters = []
                for i, parameters in enumerate(sampled_parameters):
                    modified_model_definition = substitute_parameters(
                        copy.deepcopy(model_definition), parameters)

                    trial_id = trials + i
                    hyperopt_parameters.append(
                        {
                            "parameters": parameters,
                            "model_definition": modified_model_definition,
                            "eval_split": self.split,
                            "data_df": data_df,
                            "data_train_df": data_train_df,
                            "data_validation_df": data_validation_df,
                            "data_test_df": data_test_df,
                            "data_csv": data_csv,
                            "data_train_csv": data_train_csv,
                            "data_validation_csv": data_validation_csv,
                            "data_test_csv": data_test_csv,
                            "data_hdf5": data_hdf5,
                            "data_train_hdf5": data_train_hdf5,
                            "data_validation_hdf5": data_validation_hdf5,
                            "data_test_hdf5": data_test_hdf5,
                            "train_set_metadata_json": train_set_metadata_json,
                            "experiment_name": f'{experiment_name}_{trial_id}',
                            "model_name": model_name,
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
                            'gpu_memory_limit': gpu_memory_limit,
                            'allow_parallel_threads': allow_parallel_threads,
                            'use_horovod': use_horovod,
                            'random_seed': random_seed,
                            'debug': debug,
                        }
                    )
                trials += len(sampled_parameters)

                if gpus is not None:
                    batch_results = pool.map(self._train_and_eval_model_gpu,
                                             hyperopt_parameters)
                else:
                    batch_results = pool.map(self._train_and_eval_model,
                                             hyperopt_parameters)

                self.hyperopt_strategy.update_batch(
                    (result["parameters"], result["metric_score"]) for result in
                    batch_results
                )

                hyperopt_results.extend(batch_results)
        finally:
            pool.close()
            pool.join()

        hyperopt_results = self.sort_hyperopt_results(hyperopt_results)
        return hyperopt_results


class FiberExecutor(HyperoptExecutor):
    num_workers = 2
    fiber_backend = "local"

    def __init__(
            self,
            hyperopt_strategy: HyperoptSamplingStrategy,
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

        HyperoptExecutor.__init__(self, hyperopt_strategy, output_feature,
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
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        train_kwargs = dict(
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
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            use_horovod=use_horovod,
            random_seed=random_seed,
            debug=debug,
        )

        train_fn = _train_and_eval_on_split_unary
        if self.resource_limits:
            train_fn = self.fiber_meta(**self.resource_limits)(train_fn)

        hyperopt_results = []
        trials = 0
        while not self.hyperopt_strategy.finished():
            sampled_parameters = self.hyperopt_strategy.sample_batch()
            metric_scores = []

            stats_batch = self.pool.map(
                train_fn,
                [
                    {
                        'model_definition': substitute_parameters(
                            copy.deepcopy(model_definition), parameters),
                        'experiment_name': f'{experiment_name}_{trials + i}',
                        **train_kwargs
                    }
                    for i, parameters in enumerate(sampled_parameters)
                ],
            )
            trials += len(sampled_parameters)

            for stats, parameters in zip(stats_batch, sampled_parameters):
                train_stats, eval_stats = stats
                metric_score = self.get_metric_score(eval_stats)
                metric_scores.append(metric_score)

                hyperopt_results.append(
                    {
                        "parameters": parameters,
                        "metric_score": metric_score,
                        "training_stats": train_stats,
                        "eval_stats": eval_stats,
                    }
                )

            self.hyperopt_strategy.update_batch(
                zip(sampled_parameters, metric_scores))

        hyperopt_results = self.sort_hyperopt_results(hyperopt_results)

        return hyperopt_results


def get_build_hyperopt_executor(executor_type):
    return get_from_registry(executor_type, executor_registry)


executor_registry = {
    "serial": SerialExecutor,
    "parallel": ParallelExecutor,
    "fiber": FiberExecutor,
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


def substitute_parameters(model_definition, parameters):
    parameters_dict = get_parameters_dict(parameters)
    for input_feature in model_definition["input_features"]:
        set_values(input_feature, input_feature["name"], parameters_dict)
    for output_feature in model_definition["output_features"]:
        set_values(output_feature, output_feature["name"], parameters_dict)
    set_values(model_definition["combiner"], "combiner", parameters_dict)
    set_values(model_definition["training"], "training", parameters_dict)
    set_values(model_definition["preprocessing"], "preprocessing",
               parameters_dict)
    return model_definition


# TODO this is duplicate code from experiment,
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
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=False,
        random_seed=default_random_seed,
        debug=False,
        **kwargs
):
    # Collect training and validation losses and metrics
    # & append it to `results`
    # ludwig_model = LudwigModel(modified_model_definition)
    (model, preprocessed_data, experiment_dir_name, train_stats,
     model_definition) = full_train(
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
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug,
    )
    (training_set, validation_set, test_set,
     train_set_metadata) = preprocessed_data
    if model_definition[TRAINING]["eval_batch_size"] > 0:
        batch_size = model_definition[TRAINING]["eval_batch_size"]
    else:
        batch_size = model_definition[TRAINING]["batch_size"]

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
        debug=debug
    )
    if not (
            skip_save_unprocessed_output and skip_save_test_predictions and skip_save_test_statistics):
        if not os.path.exists(experiment_dir_name):
            os.makedirs(experiment_dir_name)

    # postprocess
    postprocessed_output = postprocess(
        test_results,
        model_definition["output_features"],
        train_set_metadata,
        experiment_dir_name,
        skip_save_unprocessed_output,
    )

    print_test_results(test_results)
    if not skip_save_test_predictions:
        save_prediction_outputs(postprocessed_output, experiment_dir_name)
    if not skip_save_test_statistics:
        save_test_statistics(test_results, experiment_dir_name)
    return train_stats, test_results


def _train_and_eval_on_split_unary(kwargs):
    """Unary function is needed by Fiber to map a list of args."""
    return train_and_eval_on_split(**kwargs)
