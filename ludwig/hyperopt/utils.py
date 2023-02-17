import copy
import dataclasses
import json
import logging
import os
import warnings
from typing import Any, Dict

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUTO,
    COMBINED,
    EXECUTOR,
    GOAL,
    GRID_SEARCH,
    HYPEROPT,
    INPUT_FEATURES,
    LOSS,
    MAX_CONCURRENT_TRIALS,
    METRIC,
    MINIMIZE,
    NAME,
    NUM_SAMPLES,
    OUTPUT_FEATURES,
    PARAMETERS,
    PREPROCESSING,
    RAY,
    SPACE,
    SPLIT,
    TYPE,
    VALIDATION,
)
from ludwig.globals import HYPEROPT_STATISTICS_FILE_NAME
from ludwig.hyperopt.results import HyperoptResults, TrialResults
from ludwig.types import HyperoptConfigDict, ModelConfigDict
from ludwig.utils.data_utils import save_json
from ludwig.utils.misc_utils import (
    get_class_attributes,
    get_from_registry,
    merge_dict,
    set_default_value,
    set_default_values,
)
from ludwig.utils.print_utils import print_boxed

logger = logging.getLogger(__name__)


def print_hyperopt_results(hyperopt_results: HyperoptResults):
    print_boxed("HYPEROPT RESULTS", print_fun=logger.info)
    for trial_results in hyperopt_results.ordered_trials:
        if not isinstance(trial_results.metric_score, str):
            logger.info(f"score: {trial_results.metric_score:.6f} | parameters: {trial_results.parameters}")
    logger.info("")


def save_hyperopt_stats(hyperopt_stats, hyperopt_dir_name):
    hyperopt_stats_fn = os.path.join(hyperopt_dir_name, HYPEROPT_STATISTICS_FILE_NAME)
    save_json(hyperopt_stats_fn, hyperopt_stats)


def load_json_value(v):
    try:
        return json.loads(v)
    except Exception as e:
        logger.warning(f"While loading json, encountered exception: {e}")
        return v


# define set containing names to return for TrialResults
TRIAL_RESULTS_NAMES_SET = {f.name for f in dataclasses.fields(TrialResults)}


def load_json_values(d):
    # ensure metric_score is a string for the json load to eliminate extraneous exception message
    d["metric_score"] = str(d["metric_score"])

    # load only data required for TrialResults
    return {k: load_json_value(v) for k, v in d.items() if k in TRIAL_RESULTS_NAMES_SET}


def should_tune_preprocessing(config):
    parameters = config[HYPEROPT][PARAMETERS]
    for param_name in parameters.keys():
        if f"{PREPROCESSING}." in param_name:
            return True
    return False


def parameter_to_dict(name, value):
    if name == ".":
        # Parameter name ".", means top-level config
        return value

    parameter_dict = {}
    curr_dict = parameter_dict
    name_list = name.split(".")
    for i, name_elem in enumerate(name_list):
        if i == len(name_list) - 1:
            curr_dict[name_elem] = value
        else:
            name_dict = curr_dict.get(name_elem, {})
            curr_dict[name_elem] = name_dict
            curr_dict = name_dict
    return parameter_dict


def feature_list_to_dict(config: ModelConfigDict) -> ModelConfigDict:
    input_features_dict = {}
    for feature in config[INPUT_FEATURES]:
        input_features_dict[feature[NAME]] = feature

    output_features_dict = {}
    for feature in config[OUTPUT_FEATURES]:
        output_features_dict[feature[NAME]] = feature

    config = copy.copy(config)
    config[INPUT_FEATURES] = input_features_dict
    config[OUTPUT_FEATURES] = output_features_dict
    return config


def feature_dict_to_list(config: ModelConfigDict) -> ModelConfigDict:
    # This works because Python dicts are order-preserving, so we do not need to
    # do anything special to map from a key in the dict to an index in a list
    input_features_list = []
    for feature in config[INPUT_FEATURES].values():
        input_features_list.append(feature)

    output_features_list = []
    for feature in config[OUTPUT_FEATURES].values():
        output_features_list.append(feature)

    config = copy.copy(config)
    config[INPUT_FEATURES] = input_features_list
    config[OUTPUT_FEATURES] = output_features_list
    return config


def substitute_parameters(
    config: ModelConfigDict,
    parameters: Dict[str, Any],
):
    """Update Ludwig config with parameters sampled from the Hyperopt sampler."""

    # Collect the sets of names for each feature grouping so we can map feature names to
    # groups
    input_feature_names = {feature[NAME] for feature in config[INPUT_FEATURES]}
    output_feature_names = {feature[NAME] for feature in config[OUTPUT_FEATURES]}

    # Features in the user config are provided as a list, but in hyperopt we reference
    # features by name, so convert temporarily to a dict to simplify the mergep process.
    config = feature_list_to_dict(config)

    # Merge parameters into the user configuration in order. As such, if there are conflicting
    # params, the later params will take precedence.
    for name, value in parameters.items():
        # User params are provided as <feature_name>.<param>, but we group input / output features
        # together during the merge to make it easier and unambiguous to convert back and forth
        # TODO(travis): we should revisit the user format here, as it silently breaks situations
        # where the user has a feature named "trainer", "combiner", etc.
        prefix = name.split(".")[0]
        if prefix in input_feature_names:
            name = f"{INPUT_FEATURES}.{name}"
        elif prefix in output_feature_names:
            name = f"{OUTPUT_FEATURES}.{name}"

        param_dict = parameter_to_dict(name, value)
        config = merge_dict(config, param_dict)

    # Now that all features have been merged, convert back to the original list format.
    config = feature_dict_to_list(config)

    return config


@DeveloperAPI
def get_num_duplicate_trials(hyperopt_config: HyperoptConfigDict) -> int:
    """Returns the number of duplicate trials that will be created.

    Duplicate trials are only created when there are grid type parameters and num_samples > 1.
    """
    num_samples = hyperopt_config[EXECUTOR].get(NUM_SAMPLES, 1)
    if num_samples == 1:
        return 0

    total_grid_search_trials = 1
    for _, param_info in hyperopt_config[PARAMETERS].items():
        if param_info.get(SPACE, None) == GRID_SEARCH:
            total_grid_search_trials *= len(param_info.get("values", []))

    num_duplicate_trials = (total_grid_search_trials * num_samples) - total_grid_search_trials
    return num_duplicate_trials


def log_warning_if_all_grid_type_parameters(hyperopt_config: HyperoptConfigDict) -> None:
    """Logs warning if all parameters have a grid type search space and num_samples > 1 since this will result in
    duplicate trials being created."""
    num_duplicate_trials = get_num_duplicate_trials(hyperopt_config)
    if num_duplicate_trials == 0:
        return

    num_samples = hyperopt_config[EXECUTOR].get(NUM_SAMPLES, 1)
    warnings.warn(
        "All hyperopt parameters in Ludwig config are using grid_search space, but number of samples "
        f"({num_samples}) is greater than 1. This will result in {num_duplicate_trials} duplicate trials being "
        "created. Consider setting `num_samples` to 1 in the hyperopt executor to prevent trial duplication.",
        RuntimeWarning,
    )


def update_hyperopt_params_with_defaults(hyperopt_params: HyperoptConfigDict) -> None:
    """Updates user's Ludwig config with default hyperopt parameters."""
    from ludwig.hyperopt.execution import executor_registry

    set_default_value(hyperopt_params, EXECUTOR, {})
    set_default_value(hyperopt_params, SPLIT, VALIDATION)
    set_default_value(hyperopt_params, "output_feature", COMBINED)
    set_default_value(hyperopt_params, METRIC, LOSS)
    set_default_value(hyperopt_params, GOAL, MINIMIZE)

    set_default_values(
        hyperopt_params[EXECUTOR],
        {TYPE: RAY, NUM_SAMPLES: 1, MAX_CONCURRENT_TRIALS: AUTO},
    )

    if hyperopt_params[EXECUTOR].get("trial_driver_resources") is None:
        hyperopt_params[EXECUTOR]["trial_driver_resources"] = {"CPU": 1, "GPU": 0}

    executor = get_from_registry(hyperopt_params[EXECUTOR][TYPE], executor_registry)
    executor_defaults = {k: v for k, v in executor.__dict__.items() if k in get_class_attributes(executor)}
    set_default_values(
        hyperopt_params[EXECUTOR],
        executor_defaults,
    )
