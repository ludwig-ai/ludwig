"""automl.py.

Driver script which:

(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training parameters, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""
import argparse
import copy
import os
import warnings
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.automl.auto_tune_config import memory_tune_config
from ludwig.automl.base_config import _create_default_config, _get_reference_configs, DatasetInfo, get_dataset_info
from ludwig.automl.utils import (
    _add_transfer_config,
    _ray_init,
    get_available_resources,
    get_model_type,
    has_imbalanced_output,
    set_output_feature_metric,
)
from ludwig.constants import (
    AUTOML_DEFAULT_IMAGE_ENCODER,
    AUTOML_DEFAULT_TABULAR_MODEL,
    AUTOML_DEFAULT_TEXT_ENCODER,
    HYPEROPT,
    IMAGE,
    TABULAR,
    TEXT,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.hyperopt.run import hyperopt
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.print_utils import print_ludwig

try:
    import dask.dataframe as dd
    import ray
    from ray.tune import ExperimentAnalysis
except ImportError:
    raise ImportError(" ray is not installed. " "In order to use auto_train please run " "pip install ludwig[ray]")


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
    user_config: Dict = None,
    random_seed: int = default_random_seed,
    use_reference_config: bool = False,
    **kwargs,
) -> AutoTrainResults:
    """Main auto train API that first builds configs for each model type (e.g. concat, tabnet, transformer). Then
    selects model based on dataset attributes. And finally runs a hyperparameter optimization experiment.

    All batch and learning rate tuning is done @ training time.

    # Inputs
    :param dataset: (str, pd.DataFrame, dd.core.DataFrame) data source to train over.
    :param target: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                        as the stopping parameter
    :param output_directory: (str) directory into which to write results, defaults to
                             current working directory.
    :param tune_for_memory: (bool) refine hyperopt search space for available
                            host / GPU memory
    :param user_config: (dict) override automatic selection of specified config items
    :param random_seed: (int, default: `42`) a random seed that will be used anywhere
                        there is a call to a random number generator, including
                        hyperparameter search sampling, as well as data splitting,
                        parameter initialization and training set shuffling
    :param use_reference_config: (bool) refine hyperopt search space by setting first
                                 search point from reference model config, if any
    :param kwargs: additional keyword args passed down to `ludwig.hyperopt.run.hyperopt`.

    # Returns
    :return: (AutoTrainResults) results containing hyperopt experiments and best model
    """
    config = create_auto_config(
        dataset, target, time_limit_s, tune_for_memory, user_config, random_seed, use_reference_config
    )
    return train_with_config(dataset, config, output_directory=output_directory, random_seed=random_seed, **kwargs)


def create_auto_config(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame, DatasetInfo],
    target: Union[str, List[str]],
    time_limit_s: Union[int, float],
    tune_for_memory: bool,
    user_config: Dict = None,
    random_seed: int = default_random_seed,
    use_reference_config: bool = False,
) -> dict:
    """Returns an auto-generated Ludwig config with the intent of training the best model on given given dataset /
    target in the given time limit.

    # Inputs
    :param dataset: (str, pd.DataFrame, dd.core.DataFrame, DatasetInfo) data source to train over.
    :param target: (str, List[str]) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                         as the stopping parameter
    :param tune_for_memory: (bool) refine hyperopt search space for available
                            host / GPU memory
    :param user_config: (dict) override automatic selection of specified config items
    :param random_seed: (int, default: `42`) a random seed that will be used anywhere
                        there is a call to a random number generator, including
                        hyperparameter search sampling, as well as data splitting,
                        parameter initialization and training set shuffling
    :param use_reference_config: (bool) refine hyperopt search space by setting first
                                 search point from reference model config, if any

    # Return
    :return: (dict) selected model configuration
    """
    default_configs, features_metadata = _create_default_config(dataset, target, time_limit_s, random_seed)
    model_config, model_category, row_count = _model_select(
        dataset, default_configs, features_metadata, user_config, use_reference_config
    )
    if tune_for_memory:
        if ray.is_initialized():
            resources = get_available_resources()  # check if cluster has GPUS
            if resources["gpu"] > 0:
                model_config, fits_in_memory = ray.get(
                    ray.remote(num_gpus=1, num_cpus=1, max_calls=1)(memory_tune_config).remote(
                        model_config, dataset, model_category, row_count
                    )
                )
            else:
                model_config, fits_in_memory = ray.get(
                    ray.remote(num_cpus=1)(memory_tune_config).remote(model_config, dataset, model_category, row_count)
                )
        else:
            model_config, fits_in_memory = memory_tune_config(model_config, dataset, model_category, row_count)
        if not fits_in_memory:
            warnings.warn(
                "AutoML with tune_for_memory enabled did not return estimation that model will fit in memory. "
                "If out-of-memory occurs, consider setting AutoML user_config to reduce model memory footprint. "
            )
    return model_config


def train_with_config(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    config: dict,
    output_directory: str = OUTPUT_DIR,
    random_seed: int = default_random_seed,
    **kwargs,
) -> AutoTrainResults:
    """Performs hyperparameter optimization with respect to the given config and selects the best model.

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param config: (dict) optional Ludwig configuration to use for training, defaults
                   to `create_auto_config`.
    :param output_directory: (str) directory into which to write results, defaults to
        current working directory.
    :param random_seed: (int, default: `42`) a random seed that will be used anywhere
                        there is a call to a random number generator, including
                        hyperparameter search sampling, as well as data splitting,
                        parameter initialization and training set shuffling
    :param kwargs: additional keyword args passed down to `ludwig.hyperopt.run.hyperopt`.

    # Returns
    :return: (AutoTrainResults) results containing hyperopt experiments and best model
    """
    _ray_init()
    model_type = get_model_type(config)
    hyperopt_results = _train(
        config, dataset, output_directory=output_directory, model_name=model_type, random_seed=random_seed, **kwargs
    )
    # catch edge case where metric_score is nan
    # TODO (ASN): Decide how we want to proceed if at least one trial has
    # completed
    for trial in hyperopt_results.ordered_trials:
        if isinstance(trial.metric_score, str) or np.isnan(trial.metric_score):
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
    features_metadata,
    user_config,
    use_reference_config: bool,
):
    """Performs model selection based on dataset or user specified model.

    Note: Current implementation returns tabnet by default for tabular datasets.
    """

    dataset_info = get_dataset_info(dataset) if not isinstance(dataset, DatasetInfo) else dataset
    fields = dataset_info.fields

    base_config = default_configs["base_config"]
    model_category = None

    # tabular dataset heuristics
    if len(fields) > 3:
        model_category = TABULAR
        base_config = merge_dict(base_config, default_configs["combiner"][AUTOML_DEFAULT_TABULAR_MODEL])

        # override combiner heuristic if explicitly provided by user
        if user_config is not None:
            if "combiner" in user_config.keys():
                model_type = user_config["combiner"]["type"]
                base_config = merge_dict(base_config, default_configs["combiner"][model_type])
    else:
        # text heuristics
        for input_feature in base_config["input_features"]:
            # default text encoder is bert
            if input_feature["type"] == TEXT:
                model_category = TEXT
                input_feature["encoder"] = AUTOML_DEFAULT_TEXT_ENCODER
                base_config = merge_dict(base_config, default_configs[TEXT][AUTOML_DEFAULT_TEXT_ENCODER])
                base_config[HYPEROPT]["executor"]["num_samples"] = 5  # set for small hyperparameter search space

            # TODO (ASN): add image heuristics
            if input_feature["type"] == IMAGE:
                model_category = IMAGE
                input_feature["encoder"] = AUTOML_DEFAULT_IMAGE_ENCODER
                base_config = merge_dict(base_config, default_configs["combiner"]["concat"])

    # override and constrain automl config based on user specified values
    if user_config is not None:
        base_config = merge_dict(base_config, user_config)

        # remove all parameters from hyperparameter search that user has
        # provided explicit values for
        hyperopt_params = copy.deepcopy(base_config["hyperopt"]["parameters"])
        for hyperopt_params in hyperopt_params.keys():
            config_section, param = hyperopt_params.split(".")[0], hyperopt_params.split(".")[1]
            if config_section in user_config.keys():
                if param in user_config[config_section]:
                    del base_config["hyperopt"]["parameters"][hyperopt_params]

    # check if any binary or category output feature has highly imbalanced minority vs majority values
    # note: check is done after any relevant user_config has been applied
    has_imbalanced_output(base_config, features_metadata)

    # if single output feature, set relevant metric and goal if not already set
    base_config = set_output_feature_metric(base_config)

    # add as initial trial in the automl search the hyperparameter settings from
    # the best model for a similar dataset and matching model type, if any.
    if use_reference_config:
        ref_configs = _get_reference_configs()
        base_config = _add_transfer_config(base_config, ref_configs)

    return base_config, model_category, dataset_info.row_count


def _train(
    config: Dict,
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame],
    output_directory: str,
    model_name: str,
    random_seed: int,
    **kwargs,
):
    hyperopt_results = hyperopt(
        config,
        dataset=dataset,
        output_directory=output_directory,
        model_name=model_name,
        random_seed=random_seed,
        skip_save_log=True,  # avoid per-step log overhead by default
        **kwargs,
    )
    return hyperopt_results


def init_config(
    dataset: str,
    target: Union[str, List[str]],
    time_limit_s: Union[int, float],
    tune_for_memory: bool,
    hyperopt: bool = False,
    output: str = None,
    random_seed: int = default_random_seed,
    use_reference_config: bool = False,
    **kwargs,
):
    config = create_auto_config(
        dataset=dataset,
        target=target,
        time_limit_s=time_limit_s,
        tune_for_memory=tune_for_memory,
        random_seed=random_seed,
        use_reference_config=use_reference_config,
    )

    if HYPEROPT in config and not hyperopt:
        del config[HYPEROPT]

    if output is None:
        print(yaml.safe_dump(config, None, sort_keys=False))
    else:
        with open(output, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)


def cli_init_config(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script initializes a valid config from a dataset.",
        prog="ludwig init_config",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="input data file path",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="target(s) to predict as output features of the model",
        action="append",
        required=False,
    )
    parser.add_argument(
        "--time_limit_s",
        type=int,
        help="time limit to train the model in seconds when using hyperopt",
        required=False,
    )
    parser.add_argument(
        "--tune_for_memory",
        type=bool,
        help="refine hyperopt search space based on available host / GPU memory",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--hyperopt",
        type=bool,
        help="include automl hyperopt config",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="seed for random number generators used in hyperopt to improve repeatability",
        required=False,
    )
    parser.add_argument(
        "--use_reference_config",
        type=bool,
        help="refine hyperopt search space by setting first search point from stored reference model config",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output initialized YAML config path",
        required=False,
    )

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("init_config", *sys_argv)

    print_ludwig("Init Config", LUDWIG_VERSION)
    init_config(**vars(args))
