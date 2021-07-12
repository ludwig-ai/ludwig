"""
automl.py

Driver script which:

(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training paramers, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""
from logging import raiseExceptions
from typing import Dict, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from ludwig.automl.base_config import create_default_config
from ludwig.hyperopt.run import hyperopt

OUTPUT_DIR = "."


def model_select(default_configs):
    """
    Performs model selection based on dataset.
    Note: Current implementation returns tabnet by default. This will be
        improved in subsequent iterations
    """
    return default_configs['tabnet'], 'tabnet'


def auto_train(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    target: str,
    time_limit_s: Union[int, float],
    output_dir: str = OUTPUT_DIR
):
    """
    Main auto train API that first builds configs for each model type
    (e.g. concat, tabnet, transformer). Then selects model based on dataset
    attributes. And finally runs a hyperparameter optimization experiment.

    All batch and learning rate tuning is done @ training time.

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param target_name: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                        as the stopping parameter

    # Returns
    :return: (str) path to best trained model
    """

    default_configs = create_default_config(dataset, target, time_limit_s)
    model_config, model_name = model_select(default_configs)
    hyperopt_results = _train(model_config, dataset,
                              output_dir, model_name=model_name)
    experiment_analysis = hyperopt_results.experiment_analysis

    # catch edge case where metric_score is nan
    # TODO (ASN): Decide how we want to proceed if at least one trial has
    # completed
    for trial in hyperopt_results.ordered_trials:
        if np.isnan(trial.metric_score):
            raise ValueError(
                "There was an error running the experiment. "
                "A trial failed to start. "
                "Consider increasing the time budget for experiment. "
            )

    autotrain_results = {
        'path_to_best_model': experiment_analysis.best_checkpoint,
        'trial_id': "_".join(experiment_analysis.best_logdir.split("/")[-1].split("_")[1:])
    }
    return autotrain_results


def _train(
    config: Dict,
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    output_dir: str,
    model_name: str
):
    hyperopt_results = hyperopt(
        config,
        dataset=dataset,
        output_directory=output_dir,
        model_name=model_name
    )
    return hyperopt_results
