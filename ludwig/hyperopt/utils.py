import copy
import dataclasses
import json
import logging
import os
from typing import Any, Dict

from ludwig.constants import HYPEROPT, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PARAMETERS, PREPROCESSING
from ludwig.globals import HYPEROPT_STATISTICS_FILE_NAME
from ludwig.hyperopt.results import HyperoptResults, TrialResults
from ludwig.utils.data_utils import save_json
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.print_utils import print_boxed

logger = logging.getLogger(__name__)


def print_hyperopt_results(hyperopt_results: HyperoptResults):
    print_boxed("HYPEROPT RESULTS", print_fun=logger.info)
    for trial_results in hyperopt_results.ordered_trials:
        if not isinstance(trial_results.metric_score, str):
            logger.info(f"score: {trial_results.metric_score:.6f} | parameters: {trial_results.parameters}")


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


def feature_list_to_dict(config: Dict[str, Any]) -> Dict[str, Any]:
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


def feature_dict_to_list(config: Dict[str, Any]) -> Dict[str, Any]:
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
    config: Dict[str, Any],
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
