from collections import OrderedDict
import copy

from ludwig.api import LudwigModel
from ludwig.utils.defaults import merge_with_defaults
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata

# maps variable search space that can be modified to minimum permissible value for the range
RANKED_MODIFIABLE_PARAM_LIST = {
    'tabnet': OrderedDict({
        'training.batch_size': 32,
        'combiner.size': 8,
        'combiner.output_size': 8,
    }),
    'concat': OrderedDict({
        'training.batch_size': 32,
        'combiner.num_fc_layers': 1,
        'combiner.fc_size': 64,
    }),
    'tabtransformer': OrderedDict({
        'training.batch_size': 32,
        'combiner.num_heads:': 4,
        'combiner.output_size': 8,
        'combiner.num_layers': 4,
        'combiner.num_fc_layers': 1,
    }),
}


def get_trainingset_metadata(config, dataset):
    (_, _, _, training_set_metadata) = preprocess_for_training(
        config,
        dataset=dataset,
        preprocessing_params=config['preprocessing'])
    return training_set_metadata


def get_machine_memory():
    # default -- asssume that memory usage on GPU is 15GB
    # what about CPU?
    # TODO (ASN): improve module to support more clever ways of extracting memory bounds
    return 1.5e+13


def compute_memory_usage(config, training_set_metadata) -> int:
    update_config_with_metadata(config, training_set_metadata)
    lm = LudwigModel.create_model(config)
    lm.get_connected_model()
    model_tensors = lm.collect_weights()
    total_size = 0
    for tnsr in model_tensors:
        total_size += tnsr[1].numpy().size
    total_bytes = total_size * 32
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
        # TODO (ASN): fix to not just work with categorical search space
        current_param_values[param] = hyperparam_search_space[param]['categories'][-1]
    return current_param_values


def memory_tune_config(config, dataset):
    fits_in_memory = False
    raw_config = merge_with_defaults(config)
    training_set_metadata = get_trainingset_metadata(raw_config, dataset)
    modified_hyperparam_search_space = copy.deepcopy(
        raw_config['hyperopt']['parameters'])
    combiner_type = copy.deepcopy(raw_config['combiner']['type'])
    params_to_modify = RANKED_MODIFIABLE_PARAM_LIST[combiner_type]
    param_list = list(params_to_modify.keys())
    current_param_values = get_new_params(
        {}, modified_hyperparam_search_space, params_to_modify)
    max_memory = get_machine_memory()

    while param_list is not None:
        # compute memory utilization
        temp_config = sub_new_params(raw_config, current_param_values)
        if compute_memory_usage(temp_config, training_set_metadata) < max_memory:
            fits_in_memory = True
            break
        # check if we have exhausted tuning of current param (e.g. we can no longer reduce the param value)
        param, min_value = param_list[0],  params_to_modify[param_list[0]]
        if param in modified_hyperparam_search_space.keys() and len(modified_hyperparam_search_space[param]['categories']) > 2:
            if modified_hyperparam_search_space[param]['categories'][-2] > min_value:
                modified_hyperparam_search_space[param][
                    'categories'] = modified_hyperparam_search_space[param]['categories'][:-1]
            else:
                param_list.pop(0)
        else:
            param_list.pop(0)

    config['hyperopt']['parameters'] = modified_hyperparam_search_space
    return config, fits_in_memory
