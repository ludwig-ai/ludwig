"""Native Optuna hyperparameter optimization executor.

Provides direct Optuna integration without requiring Ray Tune as an intermediary.
Runs trials sequentially on the local machine using Ludwig's standard training API.

Supports AutoSampler (auto-selects best algorithm), GPSampler (Bayesian optimization),
TPE, CMA-ES, and other Optuna samplers.

For distributed execution, use the Ray executor with OptunaSearch instead.

Usage in Ludwig config:
    hyperopt:
      executor:
        type: optuna
        num_samples: 50
        sampler: auto  # auto, gp, tpe, cmaes, random
"""

import copy
import logging
import os
import traceback
from typing import Any

from ludwig.api import LudwigModel
from ludwig.constants import MAXIMIZE, TEST, TRAINING, VALIDATION
from ludwig.hyperopt.results import HyperoptResults, TrialResults
from ludwig.hyperopt.utils import substitute_parameters
from ludwig.utils.defaults import default_random_seed

logger = logging.getLogger(__name__)


def _create_sampler(sampler_type: str):
    """Create an Optuna sampler from type string."""
    import optuna

    if sampler_type == "auto":
        try:
            return optuna.samplers.AutoSampler()
        except AttributeError:
            logger.info("AutoSampler not available, falling back to TPE")
            return optuna.samplers.TPESampler()
    elif sampler_type == "gp":
        try:
            return optuna.samplers.GPSampler()
        except AttributeError:
            logger.info("GPSampler not available, falling back to TPE")
            return optuna.samplers.TPESampler()
    elif sampler_type == "tpe":
        return optuna.samplers.TPESampler()
    elif sampler_type == "cmaes":
        return optuna.samplers.CmaEsSampler()
    elif sampler_type == "random":
        return optuna.samplers.RandomSampler()
    else:
        raise ValueError(f"Unknown sampler: {sampler_type}. Options: auto, gp, tpe, cmaes, random")


def _suggest_params(trial, parameters: dict) -> dict[str, Any]:
    """Suggest parameter values for a trial based on the search space definition."""
    params = {}
    for param_name, space_def in parameters.items():
        space_type = space_def.get("space", "uniform")
        if space_type == "uniform":
            params[param_name] = trial.suggest_float(param_name, space_def["lower"], space_def["upper"])
        elif space_type == "loguniform":
            params[param_name] = trial.suggest_float(param_name, space_def["lower"], space_def["upper"], log=True)
        elif space_type in ("int", "randint", "qrandint"):
            params[param_name] = trial.suggest_int(param_name, int(space_def["lower"]), int(space_def["upper"]))
        elif space_type in ("choice", "categorical"):
            params[param_name] = trial.suggest_categorical(param_name, space_def["categories"])
        elif space_type == "grid_search":
            params[param_name] = trial.suggest_categorical(param_name, space_def["values"])
        else:
            raise ValueError(f"Unknown search space type: {space_type} for parameter {param_name}")
    return params


class OptunaExecutor:
    """Native Optuna hyperparameter optimization executor.

    Runs trials sequentially on the local machine. Each trial trains a full Ludwig model with parameters suggested by
    Optuna, then reports the validation metric back.
    """

    def __init__(
        self,
        parameters: dict,
        output_feature: str,
        metric: str,
        goal: str,
        split: str,
        search_alg: dict | None = None,
        num_samples: int = 10,
        sampler: str = "auto",
        pruner: str | None = None,
        study_name: str | None = None,
        storage: str | None = None,
        **kwargs,
    ) -> None:
        try:
            import optuna  # noqa: F401
        except ImportError:
            raise ImportError("Optuna is required for the optuna executor. Install with: pip install optuna")

        self.parameters = parameters
        self.output_feature = output_feature
        self.metric = metric
        self.goal = goal
        self.split = split
        self.num_samples = num_samples
        self.sampler_type = sampler
        self.pruner_type = pruner
        self.study_name = study_name or "ludwig_hyperopt"
        self.storage = storage

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
        resume=None,
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
        tune_callbacks=None,
        backend=None,
        random_seed=default_random_seed,
        debug=False,
        hyperopt_log_verbosity=3,
        **kwargs,
    ) -> HyperoptResults:
        import optuna

        sampler_obj = _create_sampler(self.sampler_type)

        pruner_obj = None
        if self.pruner_type == "median":
            pruner_obj = optuna.pruners.MedianPruner()
        elif self.pruner_type == "hyperband":
            pruner_obj = optuna.pruners.HyperbandPruner()

        direction = "minimize" if self.goal != MAXIMIZE else "maximize"
        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=sampler_obj,
            pruner=pruner_obj,
            storage=self.storage,
            load_if_exists=True,
        )

        trial_results = []

        output_dir = os.path.join(output_directory, experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        def objective(trial):
            sampled_params = _suggest_params(trial, self.parameters)

            # Substitute sampled parameters into config
            trial_config = copy.deepcopy(config)
            substitute_parameters(trial_config, sampled_params)

            trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)

            try:
                model = LudwigModel(
                    config=trial_config,
                    backend=backend,
                    gpus=gpus,
                    gpu_memory_limit=gpu_memory_limit,
                    allow_parallel_threads=allow_parallel_threads,
                    callbacks=callbacks,
                )

                train_stats, preprocessed_data, output_directory_trial = model.train(
                    dataset=dataset,
                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    training_set_metadata=training_set_metadata,
                    data_format=data_format,
                    experiment_name=f"trial_{trial.number}",
                    model_name=model_name,
                    skip_save_training_description=skip_save_training_description,
                    skip_save_training_statistics=skip_save_training_statistics,
                    skip_save_model=skip_save_model,
                    skip_save_progress=skip_save_progress,
                    skip_save_log=skip_save_log,
                    skip_save_processed_input=skip_save_processed_input,
                    output_directory=trial_dir,
                    random_seed=random_seed + trial.number,
                )

                # Evaluate on the target split
                eval_split = self.split
                eval_dataset = None
                if eval_split == TRAINING:
                    eval_dataset = preprocessed_data[0]
                elif eval_split == VALIDATION:
                    eval_dataset = preprocessed_data[1]
                elif eval_split == TEST:
                    eval_dataset = preprocessed_data[2]

                eval_stats = {}
                if eval_dataset is not None:
                    eval_stats_list, _, _ = model.evaluate(
                        dataset=eval_dataset,
                        skip_save_unprocessed_output=True,
                        skip_save_predictions=True,
                        skip_save_eval_stats=True,
                        callbacks=callbacks,
                    )
                    eval_stats = eval_stats_list

                # Extract the target metric
                metric_value = None
                if self.output_feature in eval_stats:
                    feature_stats = eval_stats[self.output_feature]
                    if self.metric in feature_stats:
                        metric_value = feature_stats[self.metric]
                elif "combined" in eval_stats and self.metric in eval_stats["combined"]:
                    metric_value = eval_stats["combined"][self.metric]

                if metric_value is None:
                    raise ValueError(
                        f"Could not find metric '{self.metric}' for output feature "
                        f"'{self.output_feature}' in evaluation stats: {list(eval_stats.keys())}"
                    )

                trial_results.append(
                    TrialResults(
                        parameters=sampled_params,
                        metric_score=metric_value,
                        training_stats=train_stats,
                        eval_stats=eval_stats,
                    )
                )

                logger.info(
                    f"Trial {trial.number}: {self.output_feature}.{self.metric} = {metric_value:.6f} "
                    f"(params: {sampled_params})"
                )

                return metric_value

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}\n{traceback.format_exc()}")
                raise optuna.TrialPruned(f"Trial failed: {e}")

        logger.info(
            f"Starting Optuna hyperopt: {self.num_samples} trials, "
            f"{self.goal} {self.output_feature}.{self.metric}, sampler={self.sampler_type}"
        )

        study.optimize(objective, n_trials=self.num_samples)

        # Sort results by metric score
        trial_results.sort(key=lambda t: t.metric_score, reverse=(self.goal == MAXIMIZE))

        logger.info(
            f"Optuna hyperopt complete. Best {self.output_feature}.{self.metric}: "
            f"{study.best_value:.6f}, params: {study.best_params}"
        )

        return HyperoptResults(ordered_trials=trial_results, experiment_analysis=study)
