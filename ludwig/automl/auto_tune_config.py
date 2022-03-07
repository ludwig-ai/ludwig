import copy
import logging
from collections import OrderedDict

import psutil

try:
    import GPUtil
except ImportError:
    raise ImportError(" ray is not installed. " "In order to use auto_train please run " "pip install ludwig[ray]")

from ludwig.api import LudwigModel
from ludwig.automl.utils import get_model_type
from ludwig.constants import (
    AUTOML_DEFAULT_TEXT_ENCODER,
    AUTOML_SMALLER_TEXT_ENCODER,
    BATCH_SIZE,
    HYPEROPT,
    PREPROCESSING,
    SPACE,
    TEXT,
    TRAINER,
)
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from ludwig.utils.defaults import merge_with_defaults
from ludwig.utils.torch_utils import initialize_pytorch

# maps variable search space that can be modified to minimum permissible value for the range
RANKED_MODIFIABLE_PARAM_LIST = {
    "tabnet": OrderedDict(
        {
            "trainer.batch_size": 32,
            "combiner.size": 8,
            "combiner.output_size": 8,
        }
    ),
    "concat": OrderedDict(
        {
            "trainer.batch_size": 32,
            "combiner.output_size": 64,
            "combiner.num_fc_layers": 1,
        }
    ),
    "tabtransformer": OrderedDict(
        {
            "trainer.batch_size": 32,
            "combiner.num_heads:": 4,
            "combiner.output_size": 8,
            "combiner.num_layers": 4,
            "combiner.num_fc_layers": 1,
        }
    ),
    "text": OrderedDict(  # for single input feature text models e.g. bert and its variants
        {
            "trainer.batch_size": 16,
        }
    ),
}


BYTES_PER_MiB = 1048576
BYTES_PER_WEIGHT = 4  # assumes 32-bit precision = 4 bytes
BYTES_OPTIMIZER_PER_WEIGHT = 8  # for optimizer m and v vectors


def get_trainingset_metadata(config, dataset):
    (_, _, _, training_set_metadata) = preprocess_for_training(
        config, dataset=dataset, preprocessing_params=config[PREPROCESSING]
    )
    return training_set_metadata


# Note: if run in Ray Cluster, this method is run remote with gpu resources requested if available
def _get_machine_memory():
    if GPUtil.getGPUs():
        machine_mem = GPUtil.getGPUs()[0].memoryTotal * BYTES_PER_MiB
    else:
        machine_mem = psutil.virtual_memory().total
    return machine_mem


def compute_memory_usage(config, training_set_metadata) -> int:
    update_config_with_metadata(config, training_set_metadata)
    lm = LudwigModel.create_model(config)
    model_size = lm.get_model_size()  # number of parameters in model
    batch_size = config[TRAINER][BATCH_SIZE]
    return model_size * (BYTES_PER_WEIGHT + BYTES_OPTIMIZER_PER_WEIGHT) * batch_size


def sub_new_params(config: dict, new_param_vals: dict):
    new_config = copy.deepcopy(config)
    for param, val in new_param_vals.items():
        config_section = param.split(".")[0]
        param_name = param.split(".")[1]
        new_config[config_section][param_name] = val
    return new_config


def get_new_params(current_param_values, hyperparam_search_space, params_to_modify):
    for param, _ in params_to_modify.items():
        if param in hyperparam_search_space:
            if hyperparam_search_space[param][SPACE] == "choice":
                current_param_values[param] = hyperparam_search_space[param]["categories"][-1]
            else:
                current_param_values[param] = hyperparam_search_space[param]["upper"]
    return current_param_values


def _reduce_text_model_size(config, training_set_metadata):
    logging.info("Text model may overflow memory; reducing model size and input sequence length")
    min_99ptile_len = float("inf")
    for feature in config["input_features"]:
        if feature["type"] == TEXT and feature["encoder"] == AUTOML_DEFAULT_TEXT_ENCODER:
            feature["encoder"] = AUTOML_SMALLER_TEXT_ENCODER
            feature_99ptile_len = training_set_metadata[feature["name"]]["word_99ptile_max_sequence_length"]
            if feature_99ptile_len < min_99ptile_len:
                min_99ptile_len = feature_99ptile_len
    if min_99ptile_len < float("inf"):
        seq_len_limit = {"word_sequence_length_limit": round(min_99ptile_len)}
        if "preprocessing" not in config:
            config["preprocessing"] = {TEXT: seq_len_limit}
        elif (
            (TEXT not in config["preprocessing"])
            or ("word_sequence_length_limit" not in config["preprocessing"][TEXT])
            or (min_99ptile_len < float(config["preprocessing"][TEXT]["word_sequence_length_limit"]))
        ):
            config["preprocessing"][TEXT] = seq_len_limit


# Note: if run in Ray Cluster, this method is run remote with gpu resources requested if available
def memory_tune_config(config, dataset, model_category):
    fits_in_memory = False
    raw_config = merge_with_defaults(config)
    training_set_metadata = get_trainingset_metadata(raw_config, dataset)
    modified_hyperparam_search_space = copy.deepcopy(raw_config[HYPEROPT]["parameters"])
    current_param_values = {}
    param_list = []
    model_type = get_model_type(raw_config)
    if model_type in RANKED_MODIFIABLE_PARAM_LIST:
        params_to_modify = RANKED_MODIFIABLE_PARAM_LIST[model_type]
        if len(params_to_modify.keys()) > 0:
            param_list = list(params_to_modify.keys())
            max_memory = _get_machine_memory()
            initialize_pytorch()

    while param_list:
        # compute memory utilization
        current_param_values = get_new_params(current_param_values, modified_hyperparam_search_space, params_to_modify)
        temp_config = sub_new_params(raw_config, current_param_values)
        mem_use = compute_memory_usage(temp_config, training_set_metadata)
        logging.info(f"Checking model estimated mem use {mem_use} against memory size {max_memory}")
        if mem_use <= max_memory:
            fits_in_memory = True
            break
        # check if we have exhausted tuning of current param (e.g. we can no longer reduce the param value)
        param, min_value = param_list[0], params_to_modify[param_list[0]]

        if param in modified_hyperparam_search_space.keys():
            param_space = modified_hyperparam_search_space[param]["space"]
            if param_space == "choice":
                if (
                    len(modified_hyperparam_search_space[param]["categories"]) >= 2
                    and modified_hyperparam_search_space[param]["categories"][-2] >= min_value
                ):
                    modified_hyperparam_search_space[param]["categories"] = modified_hyperparam_search_space[param][
                        "categories"
                    ][:-1]
                else:
                    param_list.pop(0)  # exhausted reduction of this parameter
            else:
                # reduce by 10%
                upper_bound, lower_bound = (
                    modified_hyperparam_search_space[param]["upper"],
                    modified_hyperparam_search_space[param]["lower"],
                )
                reduction_val = (upper_bound - lower_bound) * 0.1
                new_upper_bound = upper_bound - reduction_val
                if (new_upper_bound) > lower_bound and new_upper_bound > min_value:
                    modified_hyperparam_search_space[param]["upper"] = new_upper_bound
                else:
                    param_list.pop(0)  # exhausted reduction of this parameter
        else:
            param_list.pop(0)  # param not in hyperopt search space

    if (not fits_in_memory) and (model_category == TEXT):
        _reduce_text_model_size(config, training_set_metadata)

    modified_config = copy.deepcopy(config)

    modified_config[HYPEROPT]["parameters"] = modified_hyperparam_search_space
    return modified_config, fits_in_memory
