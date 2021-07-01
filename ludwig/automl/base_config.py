"""
config.py

Builds base configuration file:

(1) infer types based on dataset 
(2) populate with 
    - default combiner parameters,
    - preprocessing parameters,
    - combiner specific default training parameters,
    - combiner specific hyperopt space
    - feature parameters
(3) add machineresources 
    (base implementation -- # CPU, # GPU)
"""

import os
from typing import Dict, List, Union

import pandas as pd
from ludwig.automl.utils import (FieldInfo, get_available_resources,
                                 get_num_tokens)
from ludwig.utils.data_utils import load_yaml

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(PATH_HERE, 'defaults')

model_defaults = {
    'concat': os.path.join(CONFIG_DIR, 'concat_config.yaml'),
    'tabnet': os.path.join(CONFIG_DIR, 'tabnet_config.yaml'),
    'transformer': os.path.join(CONFIG_DIR, 'transformer_config.yaml')
}


def allocate_experiment_resources(resources: Dict) -> Dict:
    """
    Allocates ray trial resources based on available resources

    # Inputs
    :param resources (Dict) specifies all available GPUs, CPUs and associated
        metadata of the machines (i.e. memory)

    # Return
    :return: (Dict) gpu and cpu resources per trial
    """
    # TODO (ASN):
    # (1) expand logic to support multiple GPUs per trial (multi-gpu training)
    # (2) add support for kubernetes namespace (if applicable)
    # (3) add support for smarter allocation based on size of GPU memory
    experiment_resources = {
        'cpu_resources_per_trial': 1
    }
    if resources['gpu'] > 0:
        experiment_resources.update({
            'gpu_resources_per_trial': 1
        })
    return experiment_resources


def create_default_config(dataset: str, target_name: str = None, time_limit_s: Union[int, float] = None):
    """
    Returns auto_train configs for three available combiner models. 
    Coordinates the following tasks:

    - extracts fields and generates list of FieldInfo objects
    - gets field metadata (i.e avg. words, total non-null entries)
    - builds input_features and output_feautures section of config
    - for each combiner, adds default training, hyperopt
    - infers resource constraints and adds gpu and cpu resource allocation per
      trial

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param target_name: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                                    as the stopping parameter

    # Return
    :return: (Dict) dictionaries contain auto train config files for all available
    combiner types

    """
    fields, row_count = get_field_info(dataset)
    input_and_output_feature_config = get_features_config(
        fields, row_count, target_name)
    resources = get_available_resources()
    experiment_resources = allocate_experiment_resources(resources)

    model_configs = {}
    for model_name, path_to_defaults in model_defaults.items():
        default_model_config = load_yaml(path_to_defaults)
        default_model_config.update(input_and_output_feature_config)
        default_model_config['hyperopt']['executor'].update(
            experiment_resources)
        default_model_config['hyperopt']['executor']['time_budget_s'] = time_limit_s
        model_configs[model_name] = default_model_config
    return model_configs


def get_field_info(dataset: str):
    """
    Constructs FeildInfo objects for each feature in dataset. These objects
    are used for downstream type inference

    # Inputs
    :param dataset: (str) filepath to dataset.

    # Return
    :return: (List[FieldInfo]) list of FieldInfo objects

    """

    # TODO (ASN): add more detailed logic for loading dataset. initial implementation
    # assumes dataset is stored as a csv file and readable by pandas
    dataframe = pd.read_csv(dataset)
    row_count = len(dataframe)
    fields = []
    for field in dataframe.columns:
        dtype = dataframe[field].dtype.name
        distinct_values = len(dataframe[field].unique())
        nonnull_values = len(dataframe[field].notnull())
        avg_words = None
        if dtype in ['str', 'string', 'object']:
            avg_words = get_num_tokens(dataframe[field])
        fields.append(
            FieldInfo(name=field, dtype=dtype,
                      distinct_values=distinct_values, nonnull_values=nonnull_values, avg_words=avg_words)
        )
    return fields, row_count


def get_features_config(
    fields: List[FieldInfo],
    row_count: int,
    target_name: str = None,
) -> dict:
    """
    Constructs FeildInfo objects for each feature in dataset. These objects
    are used for downstream type inference

    # Inputs
    :param dataset: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param target_name (str) name of target feature

    # Return
    :return: (Dict) section of auto_train config for input_features and output_features 
    """
    metadata = get_field_metadata(fields, row_count, target_name)
    return get_config_from_metadata(metadata, target_name)


def get_config_from_metadata(metadata: list, target_name: str = None) -> dict:
    """
    Builds input/output feature sections of auto-train config using field
    metadata

    # Inputs
    :param metadata: (List[Dict]) field descriptions
    :param target_name (str) name of target feature

    # Return
    :return: (Dict) section of auto_train config for input_features and output_features
    """
    config = {
        "input_features": [],
        "output_features": [],
    }

    for field_meta in metadata:
        if field_meta["name"] == target_name:
            config["output_features"].append(field_meta["config"])
        elif not field_meta["excluded"] and field_meta["mode"] == "input":
            config["input_features"].append(field_meta["config"])

    return config


def get_field_metadata(
    fields: List[FieldInfo], row_count: int, target_name: str = None
) -> list:
    """
    Computes metadata for each field in dataset

    # Inputs
    :param dataset: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param target_name (str) name of target feature

    # Return
    :return: (List) List of dictionaries containing metadata for each field
    """

    metadata = []
    for field in fields:
        missing_value_percent = 1 - float(field.nonnull_values) / row_count
        dtype = infer_type(field, missing_value_percent, target_name)
        metadata.append(
            {
                "name": field.name,
                "config": {
                    "name": field.name,
                    "column": field.name,
                    "type": dtype,
                },
                "excluded": should_exclude(field, row_count, target_name),
                "mode": infer_mode(field, target_name),
                "missing_values": missing_value_percent,
            }
        )

    # Count of number of initial nonptext input features in the config, -1 for output
    input_count = (
        sum(
            not meta["excluded"]
            and meta["mode"] == "input"
            and meta["config"]["type"] != "text"
            for meta in metadata
        )
        - 1
    )

    # TODO (ASN): clarify logic below

    # Second pass to exclude fields that are too expensive given the constraints
    for meta in metadata:
        if input_count > 2 and meta["config"]["type"] == "text":
            # By default, exclude text inputs when there are other candidate inputs
            meta["excluded"] = True

    return metadata


def infer_type(
    field: FieldInfo, missing_value_percent: float, target_name: str = None
) -> str:
    """
    Perform type inference on field

    # Inputs
    :param dataset: (FieldInfo) object describing field
    :param missing_value_percent: (int) percentage of missing values
    :param target_name (str) name of target feature

    # Return
    :return: (str) feature type
    """
    distinct_values = field.distinct_values
    if distinct_values == 2 and (
        missing_value_percent == 0 or field.name == target_name
    ):
        return "binary"

    if distinct_values < 20:
        # TODO (tgaddair): come up with something better than this, maybe attempt to fit to Gaussian
        # NOTE (ASN): edge case -- there are less than 20 samples in dataset
        return "category"

    # add criteria for number of spaces
    if field.avg_words and field.avg_words > 2:
        return "text"

    # TODO (ASN): add other modalities (image, etc. )

    return "numerical"


def should_exclude(field: FieldInfo, row_count: int, target_name: str) -> bool:
    if field.key == "PRI":
        return True

    if field.name == target_name:
        return False

    distinct_value_percent = float(field.distinct_values) / row_count
    if distinct_value_percent == 1.0:
        upper_name = field.name.upper()
        if upper_name == "ID" or upper_name.endswith("_ID"):
            return True

    return False


def infer_mode(field: FieldInfo, target_name: str = None) -> str:
    if field.name == target_name:
        return "output"
    if field.name.lower() == "split":
        return "split"
    return "input"
