import copy
import inspect

import ray
from ray import tune

from ludwig.constants import MAXIMIZE, MINIMIZE, VALIDATION, LOSS, COMBINED
from ludwig.utils.misc_utils import set_default_value
from ludwig.hyperopt.execution import run_experiment, substitute_parameters
from ludwig.utils.defaults import default_random_seed


def get_tune_search_space(parameters):
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
    return config


def update_tune_hyperopt_params_with_defaults(hyperopt_params):
    set_default_value(hyperopt_params, "split", VALIDATION)
    set_default_value(hyperopt_params, "output_feature", COMBINED)
    set_default_value(hyperopt_params, "metric", LOSS)
    set_default_value(hyperopt_params, "goal", MINIMIZE)


class RayTuneExecutor:
    def __init__(self, search_space, output_feature, metric, split, goal, **kwargs):
        ray.init(ignore_reinit_error=True)
        self.search_space = search_space
        self.output_feature = output_feature
        self.metric = metric
        self.split = split
        self.goal = goal
        self.trial_id = 0
        self.experiment_kwargs = {}

    def _run_experiment(self, config):

        self.trial_id += 1
        hyperopt_dict = copy.deepcopy(self.experiment_kwargs)
        modified_config = substitute_parameters(
            copy.deepcopy(hyperopt_dict["config"]), config)
        hyperopt_dict["config"] = modified_config
        hyperopt_dict["experiment_name"] = f'{hyperopt_dict["experiment_name"]}_{self.trial_id}'

        train_stats, eval_stats = run_experiment(**hyperopt_dict)
        metric_score = eval_stats[self.output_feature][self.metric]

        tune.report(parameters=str(config), metric_score=metric_score,
                    training_stats=str(train_stats), eval_stats=str(eval_stats))

    def execute(self,
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
                skip_save_processed_input=False,
                skip_save_unprocessed_output=False,
                skip_save_predictions=False,
                skip_save_eval_stats=False,
                output_directory="results",
                gpus=None,
                gpu_memory_limit=None,
                allow_parallel_threads=True,
                use_horovod=None,
                random_seed=default_random_seed,
                debug=False,
                **kwargs):

        self.experiment_kwargs = dict(
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
            use_horovod=use_horovod,
            random_seed=random_seed,
            debug=debug,
        )

        analysis = tune.run(self._run_experiment, config=self.search_space)

        hyperopt_results = analysis.results_df.sort_values(
            "metric_score", ascending=self.goal != MAXIMIZE)

        return hyperopt_results.to_dict(orient="records")
