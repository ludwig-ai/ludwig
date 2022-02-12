import copy
from collections import OrderedDict

import psutil
import ray

try:
    import GPUtil
except ImportError:
    raise ImportError(" ray is not installed. " "In order to use auto_train please run " "pip install ludwig[ray]")

from ludwig.api import LudwigModel
from ludwig.automl.utils import get_available_resources, get_model_name
from ludwig.constants import BATCH_SIZE, HYPEROPT, PREPROCESSING, SPACE, TRAINER
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from ludwig.utils.defaults import merge_with_defaults

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
}


BYTES_PER_MiB = 1048576


def get_trainingset_metadata(config, dataset):
    (_, _, _, training_set_metadata) = preprocess_for_training(
        config, dataset=dataset, preprocessing_params=config[PREPROCESSING]
    )
    return training_set_metadata


def get_machine_memory():

    if ray.is_initialized():  # using ray cluster

        @ray.remote(num_gpus=1)
        def get_remote_gpu():
            gpus = GPUtil.getGPUs()
            total_mem_mb = gpus[0].memoryTotal
            return total_mem_mb * BYTES_PER_MiB

        @ray.remote(num_cpus=1)
        def get_remote_cpu():
            total_mem = psutil.virtual_memory().total
            return total_mem

        resources = get_available_resources()  # check if cluster has GPUS

        if resources["gpu"] > 0:
            machine_mem = ray.get(get_remote_gpu.remote())
        else:
            machine_mem = ray.get(get_remote_cpu.remote())
    else:  # not using ray cluster
        if GPUtil.getGPUs():
            machine_mem = GPUtil.getGPUs()[0].memoryTotal * BYTES_PER_MiB
        else:
            machine_mem = psutil.virtual_memory().total

    return machine_mem


def compute_memory_usage(config, training_set_metadata) -> int:
    update_config_with_metadata(config, training_set_metadata)
    lm = LudwigModel.create_model(config)
    model_tensors = lm.collect_weights()
    total_size = 0
    batch_size = config[TRAINER][BATCH_SIZE]
    for tnsr in model_tensors:
        total_size += tnsr[1].detach().numpy().size * batch_size
    total_bytes = total_size * 32  # assumes 32-bit precision
    return total_bytes


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


def memory_tune_config(config, dataset):
    fits_in_memory = False
    raw_config = merge_with_defaults(config)
    training_set_metadata = get_trainingset_metadata(raw_config, dataset)
    modified_hyperparam_search_space = copy.deepcopy(raw_config[HYPEROPT]["parameters"])
    params_to_modify = RANKED_MODIFIABLE_PARAM_LIST[get_model_name(raw_config)]
    param_list = list(params_to_modify.keys())
    current_param_values = {}
    max_memory = get_machine_memory()

    while param_list is not None and len(param_list) > 0:
        # compute memory utilization
        current_param_values = get_new_params(current_param_values, modified_hyperparam_search_space, params_to_modify)
        temp_config = sub_new_params(raw_config, current_param_values)
        if compute_memory_usage(temp_config, training_set_metadata) < max_memory:
            fits_in_memory = True
            break
        # check if we have exhausted tuning of current param (e.g. we can no longer reduce the param value)
        param, min_value = param_list[0], params_to_modify[param_list[0]]

        if param in modified_hyperparam_search_space.keys():
            param_space = modified_hyperparam_search_space[param]["space"]
            if param_space == "choice":
                if (
                    len(modified_hyperparam_search_space[param]["categories"]) > 2
                    and modified_hyperparam_search_space[param]["categories"][-2] > min_value
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

    modified_config = copy.deepcopy(config)

    modified_config[HYPEROPT]["parameters"] = modified_hyperparam_search_space
    return modified_config, fits_in_memory
