"""
automl.py

Driver script which:

(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training paramers, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""
from typing import Dict, Union
import warnings

import numpy as np
import pandas as pd

from ludwig.api import LudwigModel
from ludwig.automl.base_config import _create_default_config, DatasetInfo
from ludwig.automl.auto_tune_config import memory_tune_config
from ludwig.automl.utils import _ray_init
from ludwig.constants import COMBINER, TYPE
from ludwig.hyperopt.run import hyperopt

try:
    import dask.dataframe as dd
    import ray
    from ray.tune import ExperimentAnalysis
except ImportError:
    raise ImportError(
        ' ray is not installed. '
        'In order to use auto_train please run '
        'pip install ludwig[ray]'
    )


OUTPUT_DIR = "."


class AutoTrainResults:
    def __init__(self, experiment_analysis: ExperimentAnalysis):
        self._experiment_analysis = experiment_analysis

    @property
    def experiment_analysis(self):
        return self._experiment_analysis

    @property
    def path_to_best_model(self) -> str:
        return self._experiment_analysis.best_checkpoint

    @property
    def best_trial_id(self) -> str:
        return self._experiment_analysis.best_trial.trial_id

    @property
    def best_model(self) -> LudwigModel:
        return LudwigModel.load(self.path_to_best_model)


def auto_train(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    target: str,
    time_limit_s: Union[int, float],
    output_directory: str = OUTPUT_DIR,
    tune_for_memory: bool = False,
    **kwargs  # add tune_for_memory as a param
) -> AutoTrainResults:
    """
    Main auto train API that first builds configs for each model type
    (e.g. concat, tabnet, transformer). Then selects model based on dataset
    attributes. And finally runs a hyperparameter optimization experiment.

    All batch and learning rate tuning is done @ training time.

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param target: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                        as the stopping parameter
    :param output_directory: (str) directory into which to write results, defaults to
                             current working directory.

    # Returns
    :return: (AutoTrainResults) results containing hyperopt experiments and best model
    """
    config = create_auto_config(
        dataset, target, time_limit_s, tune_for_memory, ** kwargs)
    return train_with_config(
        dataset,
        config,
        output_directory=output_directory,
        **kwargs
    )


def create_auto_config(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame, DatasetInfo],
    target: str,
    time_limit_s: Union[int, float],
    tune_for_memory: bool,
    **kwargs
) -> dict:
    """
    Returns an auto-generated Ludwig config with the intent of training
    the best model on given given dataset / target in the given time
    limit.

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param target: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                                    as the stopping parameter

    # Return
    :return: (dict) selected model configuration
    """
    default_configs = _create_default_config(dataset, target, time_limit_s)
    model_config = _model_select(default_configs)
    if tune_for_memory:
        model_config, _ = memory_tune_config(model_config, dataset)
    return model_config


def train_with_config(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    config: dict,
    output_directory: str = OUTPUT_DIR,
    **kwargs,
) -> AutoTrainResults:
    """
    Performs hyperparameter optimization with respect to the given config
    and selects the best model.

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param config: (dict) optional Ludwig configuration to use for training, defaults
                   to `create_auto_config`.
    :param output_directory: (str) directory into which to write results, defaults to
        current working directory.

    # Returns
    :return: (AutoTrainResults) results containing hyperopt experiments and best model
    """
    _ray_init()
    model_name = config[COMBINER][TYPE]
    hyperopt_results = _train(
        config,
        dataset,
        output_directory=output_directory,
        model_name=model_name,
        **kwargs
    )
    # catch edge case where metric_score is nan
    # TODO (ASN): Decide how we want to proceed if at least one trial has
    # completed
    for trial in hyperopt_results.ordered_trials:
        if np.isnan(trial.metric_score):
            warnings.warn(
                "There was an error running the experiment. "
                "A trial failed to start. "
                "Consider increasing the time budget for experiment. "
            )

    experiment_analysis = hyperopt_results.experiment_analysis
    return AutoTrainResults(experiment_analysis)


def _model_select(default_configs):
    """
    Performs model selection based on dataset.
    Note: Current implementation returns tabnet by default. This will be
        improved in subsequent iterations
    """
    return default_configs['tabnet']


def _train(
    config: Dict,
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    output_directory: str,
    model_name: str,
    **kwargs
):
    hyperopt_results = hyperopt(
        config,
        dataset=dataset,
        output_directory=output_directory,
        model_name=model_name,
        backend='local',
        **kwargs
    )
    return hyperopt_results
