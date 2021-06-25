"""
config.py

Build base config
(1) infer types based on dataset (i.e. collects features)
(2) populate with 
    - default combiner parameters,
    - preprocessing parameters,
    - combiner specific default training parameters,
    - combiner specific hyperopt space
    - feature parameters
(3) add machine / resources -- how are we to know what resources the usre has -- 
    add logic to infer this (if necessary)
    (base implementation -- # CPU, # GPU)

"""

from typing import List, Dict, Union

import pandas as pd
from ludwig.automl.utils import FieldInfo, get_avg_words, get_available_resources
from ludwig.utils.data_utils import load_yaml
import os
from pathlib import Path


model_defaults = {
    'concat': os.path.join(os.path.realpath(__file__).rsplit('/', 1)[0], 'defaults/concat_config.yaml'),
    'tabnet': os.path.join(os.path.realpath(__file__).rsplit('/', 1)[0], 'defaults/tabnet_config.yaml'),
    'transformer': os.path.join(os.path.realpath(__file__).rsplit('/', 1)[0], 'defaults/transformer_config.yaml')
}


def allocate_experiment_resources(resources: Dict) -> Dict:
    # TODO (ASN):
    # (1) expand logic to support multiple GPUs per trial (multi-gpu training)
    # (2) add support for kubernetes namespace (if applicable)
    experiment_resources = {
        'cpu_resources_per_trial': '1'
    }
    if resources['gpu'] > 0:
        experiment_resources.update({
            'gpu_resources_per_trial': '1'
        })
    return experiment_resources


def create_default_config(dataset: str, target_name: str = None, time_limit_s: Union[int, float] = None):
    """
    # (1) extract fields and generate list of FieldInfo objects
    # (2) get field metadata
    # (3) build input_features and output_feautures portion of config
    # (4) for each combiner -- add default training, hyperopt
    # (5) infer resource constraints and add to hyperopt executor
    """
    fields, row_count = get_field_info(dataset)
    input_and_output_feature_config = get_input_and_output_features(
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
            avg_words = get_avg_words(dataframe[field])
        fields.append(
            FieldInfo(name=field, dtype=dtype,
                      distinct_values=distinct_values, nonnull_values=nonnull_values, avg_words=avg_words)
        )
    return fields, row_count


def get_input_and_output_features(
    fields: List[FieldInfo],
    row_count: int,
    target_name: str = None,
) -> dict:
    metadata = get_field_metadata(fields, row_count, target_name)
    return get_config_from_metadata(metadata, target_name)


def get_config_from_metadata(metadata: list, target_name: str = None) -> dict:
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
    metadata = []
    for field in fields:
        missing_value_percent = 1 - float(field.nonnull_values) / row_count
        dtype = get_predicted_type(field, missing_value_percent, target_name)
        metadata.append(
            {
                "name": field.name,
                "config": {
                    "name": field.name,
                    "column": field.name,
                    "type": dtype,
                },
                "excluded": should_exclude(field, row_count, target_name),
                "mode": get_predicted_mode(field, target_name),
                "missingValues": missing_value_percent,
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

    # Second pass to exclude fields that are too expensive given the constraints
    for meta in metadata:
        if input_count > 2 and meta["config"]["type"] == "text":
            # By default, exclude text inputs when there are other candidate inputs
            meta["excluded"] = True

    return metadata


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


def get_predicted_mode(field: FieldInfo, target_name: str = None) -> str:
    if field.name == target_name:
        return "output"
    if field.name.lower() == "split":
        return "split"
    return "input"


def get_predicted_type(
    field: FieldInfo, missing_value_percent: float, target_name: str = None
) -> str:
    distinct_values = field.distinct_values
    if distinct_values == 2 and (
        missing_value_percent == 0 or field.name == target_name
    ):
        return "binary"

    if distinct_values < 20:
        # TODO(tgaddair): come up with something better than this, maybe attempt to fit to Gaussian
        # NOTE (ASN): edge case -- there are less than 20 samples in dataset
        return "category"

    # add criteria for number of spaces
    if field.avg_words > 2:
        return "text"

    # TODO (ASN): add other modalities (image, etc. )

    return "numerical"
