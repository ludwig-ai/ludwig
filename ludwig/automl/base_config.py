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
from dataclasses import dataclass
from dataclasses_json import LetterCase, dataclass_json
from typing import List, Union

import pandas as pd

from ludwig.automl.data_source import DataSource, DataframeSource
from ludwig.automl.utils import (
    FieldInfo, get_available_resources, _ray_init, FieldMetadata, FieldConfig)
from ludwig.constants import BINARY, CATEGORY, CONFIG, IMAGE, NUMERICAL, TEXT, TYPE
from ludwig.utils.data_utils import load_yaml, load_dataset
from ludwig.utils import strings_utils

try:
    import dask.dataframe as dd
    import ray
except ImportError:
    raise ImportError(
        ' ray is not installed. '
        'In order to use auto_train please run '
        'pip install ludwig[ray]'
    )

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(PATH_HERE, 'defaults')

model_defaults = {
    'concat': os.path.join(CONFIG_DIR, 'concat_config.yaml'),
    'tabnet': os.path.join(CONFIG_DIR, 'tabnet_config.yaml'),
    'transformer': os.path.join(CONFIG_DIR, 'transformer_config.yaml')
}


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DatasetInfo:
    fields: List[FieldInfo]
    row_count: int


def allocate_experiment_resources(resources: dict) -> dict:
    """
    Allocates ray trial resources based on available resources

    # Inputs
    :param resources (dict) specifies all available GPUs, CPUs and associated
        metadata of the machines (i.e. memory)

    # Return
    :return: (dict) gpu and cpu resources per trial
    """
    # TODO (ASN):
    # (1) expand logic to support multiple GPUs per trial (multi-gpu training)
    # (2) add support for kubernetes namespace (if applicable)
    # (3) add support for smarter allocation based on size of GPU memory
    experiment_resources = {
        'cpu_resources_per_trial': 1
    }
    gpu_count, cpu_count = resources['gpu'], resources['cpu']
    if gpu_count > 0:
        experiment_resources.update({
            'gpu_resources_per_trial': 1
        })
        if cpu_count > 1:
            cpus_per_trial = max(int(cpu_count / gpu_count), 1)
            experiment_resources['cpu_resources_per_trial'] = cpus_per_trial

    return experiment_resources


def _create_default_config(
    dataset: Union[str, dd.core.DataFrame, pd.DataFrame, DatasetInfo],
    target_name: str = None,
    time_limit_s: Union[int, float] = None
) -> dict:
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
    :return: (dict) dictionaries contain auto train config files for all available
    combiner types

    """
    _ray_init()
    resources = get_available_resources()
    experiment_resources = allocate_experiment_resources(resources)

    dataset_info = dataset
    if not isinstance(dataset, DatasetInfo):
        dataset_info = get_dataset_info(dataset)

    input_and_output_feature_config = get_features_config(
        dataset_info.fields,
        dataset_info.row_count,
        resources,
        target_name
    )

    model_configs = {}
    for model_name, path_to_defaults in model_defaults.items():
        default_model_config = load_yaml(path_to_defaults)
        default_model_config.update(input_and_output_feature_config)
        default_model_config['hyperopt']['executor'].update(
            experiment_resources)
        default_model_config['hyperopt']['executor']['time_budget_s'] = time_limit_s
        model_configs[model_name] = default_model_config
    return model_configs


def get_dataset_info(dataset: str) -> DatasetInfo:
    """
    Constructs FieldInfo objects for each feature in dataset. These objects
    are used for downstream type inference

    # Inputs
    :param dataset: (str) filepath to dataset.

    # Return
    :return: (List[FieldInfo]) list of FieldInfo objects

    """
    dataframe = load_dataset(dataset)
    source = DataframeSource(dataframe)
    return get_dataset_info_from_source(source)


def get_dataset_info_from_source(source: DataSource) -> DatasetInfo:
    row_count = len(source)
    fields = []
    for field in source.columns:
        dtype = source.get_dtype(field)
        distinct_values = source.get_distinct_values(field)
        num_distinct_values = source.get_num_distinct_values(field)
        nonnull_values = source.get_nonnull_values(field)
        image_values = source.get_image_values(field)
        avg_words = None
        if source.is_string_type(dtype):
            avg_words = source.get_avg_num_tokens(field)
        fields.append(
            FieldInfo(
                name=field,
                dtype=dtype,
                distinct_values=distinct_values,
                num_distinct_values=num_distinct_values,
                nonnull_values=nonnull_values,
                image_values=image_values,
                avg_words=avg_words
            )
        )
    return DatasetInfo(fields=fields, row_count=row_count)


def get_features_config(
    fields: List[FieldInfo],
    row_count: int,
    resources: dict,
    target_name: str = None,
) -> dict:
    """
    Constructs FieldInfo objects for each feature in dataset. These objects
    are used for downstream type inference

    # Inputs
    :param dataset: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param target_name (str) name of target feature

    # Return
    :return: (dict) section of auto_train config for input_features and output_features 
    """
    metadata = get_field_metadata(fields, row_count, resources, target_name)
    return get_config_from_metadata(metadata, target_name)


def get_config_from_metadata(metadata: List[FieldMetadata], target_name: str = None) -> dict:
    """
    Builds input/output feature sections of auto-train config using field
    metadata

    # Inputs
    :param metadata: (List[FieldMetadata]) field descriptions
    :param target_name (str) name of target feature

    # Return
    :return: (dict) section of auto_train config for input_features and output_features
    """
    config = {
        "input_features": [],
        "output_features": [],
    }

    for field_meta in metadata:
        if field_meta.name == target_name:
            config["output_features"].append(field_meta.config.to_dict())
        elif not field_meta.excluded and field_meta.mode == "input":
            config["input_features"].append(field_meta.config.to_dict())

    return config


def get_field_metadata(
    fields: List[FieldInfo], row_count: int, resources: dict, target_name: str = None
) -> List[FieldMetadata]:
    """
    Computes metadata for each field in dataset

    # Inputs
    :param fields: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param target_name (str) name of target feature

    # Return
    :return: (List[FieldMetadata]) list of objects containing metadata for each field
    """

    metadata = []
    for idx, field in enumerate(fields):
        missing_value_percent = 1 - float(field.nonnull_values) / row_count
        dtype = infer_type(field, missing_value_percent)
        metadata.append(
            FieldMetadata(
                name=field.name,
                config=FieldConfig(
                    name=field.name,
                    column=field.name,
                    type=dtype,
                ),
                excluded=should_exclude(
                    idx, field, dtype, row_count, target_name),
                mode=infer_mode(field, target_name),
                missing_values=missing_value_percent,
            )
        )

    # Count of number of initial non-text input features in the config, -1 for output
    input_count = (
        sum(
            not meta.excluded
            and meta.mode == "input"
            and meta.config.type != TEXT
            for meta in metadata
        )
        - 1
    )

    # Exclude text fields if no GPUs are available
    if resources['gpu'] == 0:
        for meta in metadata:
            if input_count > 2 and meta.config.type == TEXT:
                # By default, exclude text inputs when there are other candidate inputs
                meta.excluded = True

    return metadata


def infer_type(
    field: FieldInfo,
    missing_value_percent: float,
) -> str:
    """
    Perform type inference on field

    # Inputs
    :param field: (FieldInfo) object describing field
    :param missing_value_percent: (float) percent of missing values in the column

    # Return
    :return: (str) feature type
    """
    num_distinct_values = field.num_distinct_values
    if num_distinct_values == 2 and missing_value_percent == 0:
        # Check that all distinct values are conventional bools.
        if strings_utils.are_conventional_bools(field.distinct_values):
            return BINARY
        return CATEGORY

    if field.image_values >= 3:
        return IMAGE

    if num_distinct_values < 20:
        # TODO (tgaddair): come up with something better than this, maybe attempt to fit to Gaussian
        # NOTE (ASN): edge case -- there are less than 20 samples in dataset
        return CATEGORY

    # add criteria for number of spaces
    if field.avg_words and field.avg_words > 2:
        return TEXT

    # TODO (ASN): add other modalities (image, etc. )

    return NUMERICAL


def should_exclude(idx: int, field: FieldInfo, dtype: str, row_count: int, target_name: str) -> bool:
    if field.key == "PRI":
        return True

    if field.name == target_name:
        return False

    distinct_value_percent = float(field.num_distinct_values) / row_count
    if distinct_value_percent == 1.0:
        upper_name = field.name.upper()
        if (idx == 0 and dtype == NUMERICAL) or upper_name.endswith("ID"):
            return True

    return False


def infer_mode(field: FieldInfo, target_name: str = None) -> str:
    if field.name == target_name:
        return "output"
    if field.name.lower() == "split":
        return "split"
    return "input"
