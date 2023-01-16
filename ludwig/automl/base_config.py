"""Uses heuristics to build ludwig configuration file:

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
from typing import Any, Dict, List, Set, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml
from dataclasses_json import dataclass_json, LetterCase

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend import Backend
from ludwig.constants import (
    COLUMN,
    COMBINER,
    ENCODER,
    EXECUTOR,
    HYPEROPT,
    INPUT_FEATURES,
    PREPROCESSING,
    SCHEDULER,
    SEARCH_ALG,
    SPLIT,
    TEXT,
    TYPE,
)
from ludwig.profiling import dataset_profile_pb2
from ludwig.profiling.dataset_profile import get_dataset_profile_proto, get_dataset_profile_view
from ludwig.types import ModelConfigDict
from ludwig.utils.automl.data_source import DataSource, wrap_data_source
from ludwig.utils.automl.field_info import FieldConfig, FieldInfo, FieldMetadata
from ludwig.utils.automl.type_inference import infer_type, should_exclude
from ludwig.utils.data_utils import load_yaml
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.system_utils import Resources

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(PATH_HERE, "defaults")

BASE_AUTOML_CONFIG = os.path.join(CONFIG_DIR, "base_automl_config.yaml")
REFERENCE_CONFIGS = os.path.join(CONFIG_DIR, "reference_configs.yaml")

combiner_defaults = {
    "concat": os.path.join(CONFIG_DIR, "combiner/concat_config.yaml"),
    "tabnet": os.path.join(CONFIG_DIR, "combiner/tabnet_config.yaml"),
    "transformer": os.path.join(CONFIG_DIR, "combiner/transformer_config.yaml"),
}

encoder_defaults = {"text": {"bert": os.path.join(CONFIG_DIR, "text/bert_config.yaml")}}

# Cap for number of distinct values to return.
MAX_DISTINCT_VALUES_TO_RETURN = 10


@DeveloperAPI
@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DatasetInfo:
    fields: List[FieldInfo]
    row_count: int
    size_bytes: int = -1


def allocate_experiment_resources(resources: Resources) -> dict:
    """Allocates ray trial resources based on available resources.

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
    experiment_resources = {"cpu_resources_per_trial": 1}
    gpu_count, cpu_count = resources.gpus, resources.cpus
    if gpu_count > 0:
        experiment_resources.update({"gpu_resources_per_trial": 1})
        if cpu_count > 1:
            cpus_per_trial = max(int(cpu_count / gpu_count), 1)
            experiment_resources["cpu_resources_per_trial"] = cpus_per_trial

    return experiment_resources


def get_resource_aware_hyperopt_config(
    experiment_resources: Dict[str, Any], time_limit_s: Union[int, float], random_seed: int
) -> Dict[str, Any]:
    """Returns a Ludwig config with the hyperopt section populated with appropriate parameters.

    Hyperopt parameters are intended to be appropriate for the given resources and time limit.
    """
    executor = experiment_resources
    executor.update({"time_budget_s": time_limit_s})
    if time_limit_s is not None:
        executor.update({SCHEDULER: {"max_t": time_limit_s}})

    return {
        HYPEROPT: {
            SEARCH_ALG: {"random_state_seed": random_seed},
            EXECUTOR: executor,
        },
    }


def _get_stratify_split_config(field_meta: FieldMetadata) -> dict:
    return {
        PREPROCESSING: {
            SPLIT: {
                TYPE: "stratify",
                COLUMN: field_meta.name,
            }
        }
    }


def get_default_automl_hyperopt() -> Dict[str, Any]:
    """Returns general, default settings for hyperopt.

    For example:
    - We set a random_state_seed for sample sequence repeatability
    - We use an increased reduction_factor to get more pruning/exploration.

    TODO: If settings seem reasonable, consider building this into the hyperopt schema, directly.
    """
    return yaml.safe_load(
        """
  search_alg:
    type: hyperopt
  executor:
    type: ray
    num_samples: 10
    time_budget_s: 3600
    scheduler:
      type: async_hyperband
      time_attr: time_total_s
      max_t: 3600
      grace_period: 72
      reduction_factor: 5
"""
    )


def create_default_config(
    features_config: ModelConfigDict,
    dataset_info: DatasetInfo,
    target_name: Union[str, List[str]],
    time_limit_s: Union[int, float],
    random_seed: int,
    imbalance_threshold: float = 0.9,
    backend: Backend = None,
) -> dict:
    """Returns auto_train configs for three available combiner models. Coordinates the following tasks:

    - extracts fields and generates list of FieldInfo objects
    - gets field metadata (i.e avg. words, total non-null entries)
    - builds input_features and output_features section of config
    - for imbalanced datasets, a preprocessing section is added to perform stratified sampling if the imbalance ratio
      is smaller than imbalance_threshold
    - for each combiner, adds default training, hyperopt
    - infers resource constraints and adds gpu and cpu resource allocation per
      trial

    # Inputs
    :param dataset_info: (str) filepath Dataset Info object.
    :param target_name: (str, List[str]) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                                    as the stopping parameter
    :param random_seed: (int, default: `42`) a random seed that will be used anywhere
                        there is a call to a random number generator, including
                        hyperparameter search sampling, as well as data splitting,
                        parameter initialization and training set shuffling
    :param imbalance_threshold: (float) maximum imbalance ratio (minority / majority) to perform stratified sampling
    :param backend: (Backend) backend to use for training.

    # Return
    :return: (dict) dictionaries contain auto train config files for all available
    combiner types
    """
    base_automl_config = load_yaml(BASE_AUTOML_CONFIG)
    base_automl_config.update(features_config)

    targets = convert_targets(target_name)
    features_metadata = get_field_metadata(dataset_info.fields, dataset_info.row_count, targets)

    # Handle expensive features for CPU
    resources = backend.get_available_resources()
    for ifeature in base_automl_config[INPUT_FEATURES]:
        if resources.gpus == 0:
            if ifeature[TYPE] == TEXT:
                # When no GPUs are available, default to the embed encoder, which is fast enough for CPU
                ifeature[ENCODER] = {"type": "embed"}

    # create set of all feature types appearing in the dataset
    feature_types = [[feat[TYPE] for feat in features] for features in features_config.values()]
    feature_types = set(sum(feature_types, []))

    model_configs = {}

    # update hyperopt config
    experiment_resources = allocate_experiment_resources(resources)
    base_automl_config = merge_dict(
        base_automl_config, get_resource_aware_hyperopt_config(experiment_resources, time_limit_s, random_seed)
    )

    # add preprocessing section if single output feature is imbalanced
    outputs_metadata = [f for f in features_metadata if f.mode == "output"]
    if len(outputs_metadata) == 1:
        of_meta = outputs_metadata[0]
        is_categorical = of_meta.config.type in ["category", "binary"]
        is_imbalanced = of_meta.imbalance_ratio < imbalance_threshold
        if is_categorical and is_imbalanced:
            base_automl_config.update(_get_stratify_split_config(of_meta))

    model_configs["base_config"] = base_automl_config

    # read in all encoder configs
    for feat_type, default_configs in encoder_defaults.items():
        if feat_type in feature_types:
            if feat_type not in model_configs.keys():
                model_configs[feat_type] = {}
            for encoder_name, encoder_config_path in default_configs.items():
                model_configs[feat_type][encoder_name] = load_yaml(encoder_config_path)

    # read in all combiner configs
    model_configs[COMBINER] = {}
    for combiner_type, default_config in combiner_defaults.items():
        combiner_config = load_yaml(default_config)
        model_configs[COMBINER][combiner_type] = combiner_config

    return model_configs


# Read in the score and configuration of a reference model trained by Ludwig for each dataset in a list.
def get_reference_configs() -> dict:
    reference_configs = load_yaml(REFERENCE_CONFIGS)
    return reference_configs


def get_dataset_info(df: Union[pd.DataFrame, dd.core.DataFrame]) -> DatasetInfo:
    """Constructs FieldInfo objects for each feature in dataset. These objects are used for downstream type
    inference.

    # Inputs
    :param df: (Union[pd.DataFrame, dd.core.DataFrame]) Pandas or Dask dataframe.

    # Return
    :return: (DatasetInfo) Structure containing list of FieldInfo objects.
    """
    source = wrap_data_source(df)
    return get_dataset_info_from_source(source)


def is_field_boolean(source: DataSource, field: str) -> bool:
    """Returns a boolean indicating whether the object field should have a bool dtype.

    Columns with object dtype that have 3 distinct values of which one is Nan/None is a bool type column.
    """
    unique_values = source.df[field].unique()
    if len(unique_values) <= 3:
        for entry in unique_values:
            try:
                if np.isnan(entry):
                    continue
            except TypeError:
                # For some field types such as object arrays, np.isnan throws a TypeError
                # In this case, do nothing and proceed to checking if the entry is a bool object
                pass
            if isinstance(entry, bool):
                continue
            return False
        return True
    return False


@DeveloperAPI
def get_dataset_profile_from_source(source: DataSource) -> dataset_profile_pb2.DatasetProfile:
    return get_dataset_profile_proto(get_dataset_profile_view(source.df))


@DeveloperAPI
def get_dataset_info_from_source(source: DataSource) -> DatasetInfo:
    """Constructs FieldInfo objects for each feature in dataset. These objects are used for downstream type
    inference.

    # Inputs
    :param source: (DataSource) A wrapper around a data source, which may represent a pandas or Dask dataframe.

    # Return
    :return: (DatasetInfo) Structure containing list of FieldInfo objects.
    """
    row_count = len(source)
    fields = []
    for field in source.columns:
        dtype = source.get_dtype(field)
        num_distinct_values, distinct_values, distinct_values_balance = source.get_distinct_values(
            field, MAX_DISTINCT_VALUES_TO_RETURN
        )
        nonnull_values = source.get_nonnull_values(field)
        image_values = source.get_image_values(field)
        audio_values = source.get_audio_values(field)
        avg_words = None
        if dtype == "object":
            # Check if it is a nullboolean field. We do this since if you read a csv with
            # pandas that has a column of booleans and some missing values, the column is
            # interpreted as object dtype instead of bool
            if is_field_boolean(source, field):
                dtype = "bool"
        if source.is_string_type(dtype):
            avg_words = source.get_avg_num_tokens(field)
        fields.append(
            FieldInfo(
                name=field,
                dtype=dtype,
                distinct_values=distinct_values,
                num_distinct_values=num_distinct_values,
                distinct_values_balance=distinct_values_balance,
                nonnull_values=nonnull_values,
                image_values=image_values,
                audio_values=audio_values,
                avg_words=avg_words,
            )
        )
    return DatasetInfo(fields=fields, row_count=row_count, size_bytes=source.size_bytes())


def get_features_config(
    fields: List[FieldInfo],
    row_count: int,
    target_name: Union[str, List[str]] = None,
) -> dict:
    """Constructs FieldInfo objects for each feature in dataset. These objects are used for downstream type
    inference.

    # Inputs
    :param fields: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param target_name (str, List[str]) name of target feature

    # Return
    :return: (dict) section of auto_train config for input_features and output_features
    """
    targets = convert_targets(target_name)
    metadata = get_field_metadata(fields, row_count, targets)
    return get_config_from_metadata(metadata, targets)


def convert_targets(target_name: Union[str, List[str]] = None) -> Set[str]:
    targets = target_name
    if isinstance(targets, str):
        targets = [targets]
    if targets is None:
        targets = []
    return set(targets)


def get_config_from_metadata(metadata: List[FieldMetadata], targets: Set[str] = None) -> dict:
    """Builds input/output feature sections of auto-train config using field metadata.

    # Inputs
    :param metadata: (List[FieldMetadata]) field descriptions
    :param targets (Set[str]) names of target features

    # Return
    :return: (dict) section of auto_train config for input_features and output_features
    """
    config = {
        "input_features": [],
        "output_features": [],
    }

    for field_meta in metadata:
        if field_meta.name in targets:
            config["output_features"].append(field_meta.config.to_dict())
        elif not field_meta.excluded and field_meta.mode == "input":
            config["input_features"].append(field_meta.config.to_dict())

    return config


@DeveloperAPI
def get_field_metadata(fields: List[FieldInfo], row_count: int, targets: Set[str] = None) -> List[FieldMetadata]:
    """Computes metadata for each field in dataset.

    # Inputs
    :param fields: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param targets (Set[str]) names of target features

    # Return
    :return: (List[FieldMetadata]) list of objects containing metadata for each field
    """

    metadata = []
    column_count = len(fields)
    for idx, field in enumerate(fields):
        missing_value_percent = 1 - float(field.nonnull_values) / row_count
        dtype = infer_type(field, missing_value_percent, row_count)
        metadata.append(
            FieldMetadata(
                name=field.name,
                config=FieldConfig(
                    name=field.name,
                    column=field.name,
                    type=dtype,
                ),
                excluded=should_exclude(idx, field, dtype, column_count, row_count, targets),
                mode=infer_mode(field, targets),
                missing_values=missing_value_percent,
                imbalance_ratio=field.distinct_values_balance,
            )
        )

    return metadata


def infer_mode(field: FieldInfo, targets: Set[str] = None) -> str:
    if field.name in targets:
        return "output"
    if field.name.lower() == "split":
        return "split"
    return "input"
