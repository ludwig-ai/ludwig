"""
automl.py

Driver script which:

(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training paramers, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""

import os
import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd
import copy

from ludwig.api import LudwigModel
from ludwig.automl.base_config import (
    _create_default_config,
    DatasetInfo,
    get_dataset_info,
    infer_type,
)
from ludwig.automl.auto_tune_config import memory_tune_config
from ludwig.automl.utils import _ray_init, get_model_name
from ludwig.constants import COMBINER, NUMERICAL, TYPE
from ludwig.hyperopt.run import hyperopt
from ludwig.utils.misc_utils import merge_dict

try:
    import dask.dataframe as dd
    import ray
    from ray.tune import ExperimentAnalysis
except ImportError:
    raise ImportError(
        " ray is not installed. "
        "In order to use auto_train please run "
        "pip install ludwig[ray]"
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
        return LudwigModel.load(os.path.join(self.path_to_best_model, "model"))


def auto_train(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    target: str,
    time_limit_s: Union[int, float],
    output_directory: str = OUTPUT_DIR,
    tune_for_memory: bool = False,
    user_specified_config: Dict = None,
    **kwargs,
) -> AutoTrainResults:
    """
    Main auto train API that first builds configs for each model type
    (e.g. concat, tabnet, transformer). Then selects model based on dataset
    attributes. And finally runs a hyperparameter optimization experiment.

    All batch and learning rate tuning is done @ training time.

    # Inputs
    :param dataset: (str, pd.DataFrame, dd.core.DataFrame) data source to train over.
    :param target: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                        as the stopping parameter
    :param output_directory: (str) directory into which to write results, defaults to
                             current working directory.

    # Returns
    :return: (AutoTrainResults) results containing hyperopt experiments and best model
    """
    config = create_auto_config(
        dataset,
        target,
        time_limit_s,
        tune_for_memory,
        user_specified_config,
        **kwargs,
    )
    return train_with_config(
        dataset, config, output_directory=output_directory, **kwargs
    )


def create_auto_config(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame, DatasetInfo],
    target: str,
    time_limit_s: Union[int, float],
    tune_for_memory: bool,
    user_specified_config: Dict = None,
) -> dict:
    """
    Returns an auto-generated Ludwig config with the intent of training
    the best model on given given dataset / target in the given time
    limit.

    # Inputs
    :param dataset: (str, pd.DataFrame, dd.core.DataFrame, DatasetInfo) data source to train over.
    :param target: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                                    as the stopping parameter

    # Return
    :return: (dict) selected model configuration
    """
    default_configs = _create_default_config(dataset, target, time_limit_s)
    model_config = _model_select(
        dataset, default_configs, user_specified_config
    )
    if tune_for_memory:
        if ray.is_initialized():
            model_config, _ = ray.get(
                ray.remote(num_cpus=1)(memory_tune_config).remote(
                    model_config, dataset
                )
            )
        else:
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
    model_name = get_model_name(config)
    hyperopt_results = _train(
        config,
        dataset,
        output_directory=output_directory,
        model_name=model_name,
        **kwargs,
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


def _model_select(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame, DatasetInfo],
    default_configs,
    user_specified_config,
):
    """
    Performs model selection based on dataset or user specified model.
    Note: Current implementation returns tabnet by default. If the
    percentage of numerical features is >90%, the concat model is used.
    """

    dataset_info = get_dataset_info(dataset) if not isinstance(
        dataset, DatasetInfo) else dataset
    fields = dataset_info.fields
    row_count = dataset_info.row_count

    total_numerical_feats = 0

    for idx, field in enumerate(fields):
        missing_value_percent = 1 - float(field.nonnull_values) / row_count
        dtype = infer_type(field, missing_value_percent)
        if dtype == NUMERICAL:
            total_numerical_feats += 1

    percent_numerical_feats = total_numerical_feats / len(fields)

    base_config = default_configs["base_config"]

    # tabular dataset heuristics
    if len(fields) > 3:
        if percent_numerical_feats > 0.9:
            base_config = (
                base_config, default_configs["combiner"]["concat"])
        else:
            base_config = merge_dict(
                base_config, default_configs["combiner"]["tabnet"])

        # override combiner heuristic if explicitly provided by user
        if user_specified_config is not None:
            if "combiner" in user_specified_config.keys():
                model_type = user_specified_config["combiner"]["type"]
                base_config = merge_dict(
                    base_config, default_configs["combiner"][model_type])
    else:
        # text heuristics
        for input_feature in base_config["input_features"]:
            # default text encoder is bert
            # TODO (ASN): add more robust heuristics
            if input_feature["type"] == "text":
                input_feature["encoder"] = "bert"
                base_config = merge_dict(
                    base_config, default_configs["text"]["bert"])

            # TODO (ASN): add image heuristics

    # override and constrain automl config based on user specified values
    if user_specified_config is not None:
        base_config = merge_dict(base_config, user_specified_config)

        # remove all parameters from hyperparameter search that user has
        # provided explicit values for
        hyperopt_params = copy.deepcopy(base_config["hyperopt"]["parameters"])
        for hyperopt_params in hyperopt_params.keys():
            config_section, param = hyperopt_params.split(
                ".")[0], hyperopt_params.split(".")[1]
            if config_section in user_specified_config.keys():
                if param in user_specified_config[config_section]:
                    del base_config["hyperopt"]["parameters"][hyperopt_params]

    return base_config


def _train(
    config: Dict,
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    output_directory: str,
    model_name: str,
    **kwargs,
):
    hyperopt_results = hyperopt(
        config,
        dataset=dataset,
        output_directory=output_directory,
        model_name=model_name,
        backend="local",
        **kwargs,
    )
    return hyperopt_results
