"""automl.py.

Driver script which:

(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training parameters, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""
import argparse
import copy
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI, PublicAPI
from ludwig.automl.auto_tune_config import memory_tune_config
from ludwig.automl.base_config import (
    _create_default_config,
    _get_reference_configs,
    allocate_experiment_resources,
    DatasetInfo,
    get_dataset_info,
    get_default_automl_hyperopt,
    get_resource_aware_hyperopt_config,
)
from ludwig.backend import Backend, initialize_backend
from ludwig.constants import (
    AUTOML_DEFAULT_IMAGE_ENCODER,
    AUTOML_DEFAULT_TABULAR_MODEL,
    AUTOML_DEFAULT_TEXT_ENCODER,
    ENCODER,
    HYPEROPT,
    IMAGE,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    TABULAR,
    TEXT,
    TYPE,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.hyperopt.run import hyperopt
from ludwig.profiling import dataset_profile_pb2
from ludwig.profiling.dataset_profile import (
    get_column_profile_summaries_from_proto,
    get_dataset_profile_proto,
    get_dataset_profile_view,
)
from ludwig.profiling.type_inference import get_ludwig_type_map_from_column_profile_summaries
from ludwig.utils.automl.ray_utils import _ray_init
from ludwig.utils.automl.utils import _add_transfer_config, get_model_type, set_output_feature_metric
from ludwig.utils.data_utils import load_dataset, use_credentials
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import open_file
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.print_utils import print_ludwig
from ludwig.utils.types import DataFrame

try:
    import dask.dataframe as dd
    import ray
    from ray.tune import ExperimentAnalysis
except ImportError as e:
    raise RuntimeError("ray is not installed. In order to use auto_train please run pip install ludwig[ray]") from e


logger = logging.getLogger(__name__)

OUTPUT_DIR = "."


class AutoTrainResults:
    def __init__(self, experiment_analysis: ExperimentAnalysis, creds: Dict[str, Any] = None):
        self._experiment_analysis = experiment_analysis
        self._creds = creds

    @property
    def experiment_analysis(self):
        return self._experiment_analysis

    @property
    def best_trial_id(self) -> str:
        return self._experiment_analysis.best_trial.trial_id

    @property
    def best_model(self) -> Optional[LudwigModel]:
        checkpoint = self._experiment_analysis.best_checkpoint
        if checkpoint is None:
            logger.warning("No best model found")
            return None

        ckpt_type, ckpt_path = checkpoint.get_internal_representation()
        if ckpt_type == "uri":
            # Read remote URIs using Ludwig's internal remote file loading APIs, as
            # Ray's do not handle custom credentials at the moment.
            with use_credentials(self._creds):
                return LudwigModel.load(os.path.join(ckpt_path, "model"))
        else:
            with checkpoint.as_directory() as ckpt_path:
                return LudwigModel.load(os.path.join(ckpt_path, "model"))


@PublicAPI
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
        dataset,
        target,
        time_limit_s,
        tune_for_memory,
        user_config,
        random_seed,
        use_reference_config=use_reference_config,
    )
    return train_with_config(dataset, config, output_directory=output_directory, random_seed=random_seed, **kwargs)


@DeveloperAPI
def create_auto_config_with_dataset_profile(
    target: str,
    dataset: Optional[Union[str, DataFrame]] = None,
    dataset_profile: dataset_profile_pb2.DatasetProfile = None,
    random_seed: int = default_random_seed,
    include_hyperopt: bool = False,
    time_limit_s: Union[int, float] = None,
    backend: Union[Backend, str] = None,
) -> dict:
    """Returns the best single-shot Ludwig config given a Ludwig dataset or dataset profile.

    If only the dataset is provided, then a new profile is computed.
    Only one of the dataset or dataset_profile should be specified, not both.

    This function is intended to eventually replace create_auto_config().
    """
    if dataset is None and dataset_profile is None:
        raise ValueError("Please specify either a dataset or a dataset_profile.")
    if dataset is not None and dataset_profile is not None:
        raise ValueError("Please specify either a dataset or a dataset_profile. It is an error to specify both.")

    # Get the dataset profile.
    if dataset_profile is None:
        dataset_profile = get_dataset_profile_proto(get_dataset_profile_view(dataset))

    # Use the dataset profile to get Ludwig types.
    ludwig_type_map = get_ludwig_type_map_from_column_profile_summaries(
        get_column_profile_summaries_from_proto(dataset_profile)
    )

    # Add features along with their profiled types.
    automl_config = {}
    automl_config[INPUT_FEATURES] = []
    automl_config[OUTPUT_FEATURES] = []
    for feature_name, ludwig_type in ludwig_type_map.items():
        if feature_name == target:
            automl_config[OUTPUT_FEATURES].append({"name": feature_name, "type": ludwig_type})
        else:
            automl_config[INPUT_FEATURES].append({"name": feature_name, "type": ludwig_type})

    # Set the combiner to tabnet, by default.
    automl_config.get("combiner", {})[TYPE] = "tabnet"

    # Add hyperopt, if desired.
    if include_hyperopt:
        automl_config[HYPEROPT] = get_default_automl_hyperopt()

        # Merge resource-sensitive settings.
        backend = initialize_backend(backend)
        resources = backend.get_available_resources()
        experiment_resources = allocate_experiment_resources(resources)
        automl_config = merge_dict(
            automl_config, get_resource_aware_hyperopt_config(experiment_resources, time_limit_s, random_seed)
        )

    # TODO: Adjust preprocessing parameters according to output feature imbalance.
    return automl_config


@PublicAPI
def create_auto_config(
    dataset: Union[str, pd.DataFrame, dd.core.DataFrame, DatasetInfo],
    target: Union[str, List[str]],
    time_limit_s: Union[int, float],
    tune_for_memory: bool,
    user_config: Dict = None,
    random_seed: int = default_random_seed,
    imbalance_threshold: float = 0.9,
    use_reference_config: bool = False,
    backend: Union[Backend, str] = None,
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
    :param imbalance_threshold: (float) maximum imbalance ratio (minority / majority) to perform stratified sampling
    :param use_reference_config: (bool) refine hyperopt search space by setting first
                                 search point from reference model config, if any

    # Return
    :return: (dict) selected model configuration
    """
    backend = initialize_backend(backend)

    if not isinstance(dataset, DatasetInfo):
        dataset = load_dataset(dataset, df_lib=backend.df_engine.df_lib)

    print("Auto-generating config for dataset: {} ...".format(dataset))

    dataset_info = get_dataset_info(dataset) if not isinstance(dataset, DatasetInfo) else dataset
    default_configs = _create_default_config(
        dataset_info, target, time_limit_s, random_seed, imbalance_threshold, backend
    )
    model_config, model_category, row_count = _model_select(
        dataset_info, default_configs, user_config, use_reference_config
    )
    if tune_for_memory:
        args = (model_config, dataset, model_category, row_count, backend)
        if ray.is_initialized():
            resources = backend.get_available_resources()  # check if cluster has GPUS
            if resources.gpus > 0:
                model_config, fits_in_memory = ray.get(
                    ray.remote(num_gpus=1, num_cpus=1, max_calls=1)(memory_tune_config).remote(*args)
                )
            else:
                model_config, fits_in_memory = ray.get(ray.remote(num_cpus=1)(memory_tune_config).remote(*args))
        else:
            model_config, fits_in_memory = memory_tune_config(*args)
        if not fits_in_memory:
            warnings.warn(
                "AutoML with tune_for_memory enabled did not return estimation that model will fit in memory. "
                "If out-of-memory occurs, consider setting AutoML user_config to reduce model memory footprint. "
            )
    return model_config


@PublicAPI
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

    # Extract credentials needed to pull artifacts, if provided
    creds = None
    backend: Backend = initialize_backend(kwargs.get("backend"))
    if backend is not None:
        creds = backend.storage.artifacts.credentials

    experiment_analysis = hyperopt_results.experiment_analysis
    return AutoTrainResults(experiment_analysis, creds)


def _model_select(
    dataset_info: DatasetInfo,
    default_configs,
    user_config,
    use_reference_config: bool,
):
    """Performs model selection based on dataset or user specified model.

    Note: Current implementation returns tabnet by default for tabular datasets.
    """
    fields = dataset_info.fields

    base_config = copy.deepcopy(default_configs["base_config"])
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
        for input_feature in default_configs["base_config"]["input_features"]:
            # default text encoder is bert
            if input_feature[TYPE] == TEXT:
                model_category = TEXT
                if ENCODER in input_feature:
                    input_feature[ENCODER][TYPE] = AUTOML_DEFAULT_TEXT_ENCODER
                else:
                    input_feature[ENCODER] = {TYPE: AUTOML_DEFAULT_TEXT_ENCODER}
                # TODO(shreya): Should this hyperopt config param be set here?
                base_config[HYPEROPT]["executor"]["num_samples"] = 5  # set for small hyperparameter search space
                base_config = merge_dict(base_config, default_configs[TEXT][AUTOML_DEFAULT_TEXT_ENCODER])

            # TODO (ASN): add image heuristics
            if input_feature[TYPE] == IMAGE:
                model_category = IMAGE
                if ENCODER in input_feature:
                    input_feature[ENCODER][TYPE] = AUTOML_DEFAULT_IMAGE_ENCODER
                else:
                    input_feature[ENCODER] = {TYPE: AUTOML_DEFAULT_IMAGE_ENCODER}

        # Merge combiner config
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
        with open_file(output, "w") as f:
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
