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
from typing import List, Set, Union

import dask.dataframe as dd
import pandas as pd
from dataclasses_json import dataclass_json, LetterCase

from ludwig.automl.data_source import DataframeSource, DataSource
from ludwig.automl.utils import _ray_init, FieldConfig, FieldInfo, FieldMetadata, get_available_resources
from ludwig.constants import AUDIO, BINARY, CATEGORY, DATE, IMAGE, NUMBER, TEXT
from ludwig.utils import strings_utils
from ludwig.utils.data_utils import load_dataset, load_yaml
from ludwig.utils.defaults import default_random_seed

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

# For a given feature, the highest percentage of distinct values out of the total number of rows that we might still
# assign the CATEGORY type.
CATEGORY_TYPE_DISTINCT_VALUE_PERCENTAGE_CUTOFF = 0.5

# Cap for number of distinct values to return.
MAX_DISTINCT_VALUES_TO_RETURN = 10


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DatasetInfo:
    fields: List[FieldInfo]
    row_count: int


def allocate_experiment_resources(resources: dict) -> dict:
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
    gpu_count, cpu_count = resources["gpu"], resources["cpu"]
    if gpu_count > 0:
        experiment_resources.update({"gpu_resources_per_trial": 1})
        if cpu_count > 1:
            cpus_per_trial = max(int(cpu_count / gpu_count), 1)
            experiment_resources["cpu_resources_per_trial"] = cpus_per_trial

    return experiment_resources


def _create_default_config(
    dataset: Union[str, dd.core.DataFrame, pd.DataFrame, DatasetInfo],
    target_name: Union[str, List[str]] = None,
    time_limit_s: Union[int, float] = None,
    random_seed: int = default_random_seed,
) -> dict:
    """Returns auto_train configs for three available combiner models. Coordinates the following tasks:

    - extracts fields and generates list of FieldInfo objects
    - gets field metadata (i.e avg. words, total non-null entries)
    - builds input_features and output_features section of config
    - for each combiner, adds default training, hyperopt
    - infers resource constraints and adds gpu and cpu resource allocation per
      trial

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param target_name: (str, List[str]) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                                    as the stopping parameter
    :param random_seed: (int, default: `42`) a random seed that will be used anywhere
                        there is a call to a random number generator, including
                        hyperparameter search sampling, as well as data splitting,
                        parameter initialization and training set shuffling

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
        dataset_info.fields, dataset_info.row_count, resources, target_name
    )
    # create set of all feature types appearing in the dataset
    feature_types = [[feat["type"] for feat in features] for features in input_and_output_feature_config.values()]
    feature_types = set(sum(feature_types, []))

    model_configs = {}

    # read in base config and update with experiment resources
    base_automl_config = load_yaml(BASE_AUTOML_CONFIG)
    base_automl_config["hyperopt"]["executor"].update(experiment_resources)
    base_automl_config["hyperopt"]["executor"]["time_budget_s"] = time_limit_s
    if time_limit_s is not None:
        base_automl_config["hyperopt"]["sampler"]["scheduler"]["max_t"] = time_limit_s
    base_automl_config["hyperopt"]["sampler"]["search_alg"]["random_state_seed"] = random_seed
    base_automl_config.update(input_and_output_feature_config)

    model_configs["base_config"] = base_automl_config

    # read in all encoder configs
    for feat_type, default_configs in encoder_defaults.items():
        if feat_type in feature_types:
            if feat_type not in model_configs.keys():
                model_configs[feat_type] = {}
            for encoder_name, encoder_config_path in default_configs.items():
                model_configs[feat_type][encoder_name] = load_yaml(encoder_config_path)

    # read in all combiner configs
    model_configs["combiner"] = {}
    for combiner_type, default_config in combiner_defaults.items():
        combiner_config = load_yaml(default_config)
        model_configs["combiner"][combiner_type] = combiner_config

    return model_configs


# Read in the score and configuration of a reference model trained by Ludwig for each dataset in a list.
def _get_reference_configs() -> dict:
    reference_configs = load_yaml(REFERENCE_CONFIGS)
    return reference_configs


def get_dataset_info(dataset: Union[str, pd.DataFrame, dd.core.DataFrame]) -> DatasetInfo:
    """Constructs FieldInfo objects for each feature in dataset. These objects are used for downstream type
    inference.

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
        num_distinct_values, distinct_values = source.get_distinct_values(field, MAX_DISTINCT_VALUES_TO_RETURN)
        nonnull_values = source.get_nonnull_values(field)
        image_values = source.get_image_values(field)
        audio_values = source.get_audio_values(field)
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
                audio_values=audio_values,
                avg_words=avg_words,
            )
        )
    return DatasetInfo(fields=fields, row_count=row_count)


def get_features_config(
    fields: List[FieldInfo],
    row_count: int,
    resources: dict,
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
    targets = target_name
    if isinstance(targets, str):
        targets = [targets]
    if targets is None:
        targets = []
    targets = set(targets)

    metadata = get_field_metadata(fields, row_count, resources, targets)
    return get_config_from_metadata(metadata, targets)


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


def get_field_metadata(
    fields: List[FieldInfo], row_count: int, resources: dict, targets: Set[str] = None
) -> List[FieldMetadata]:
    """Computes metadata for each field in dataset.

    # Inputs
    :param fields: (List[FieldInfo]) FieldInfo objects for all fields in dataset
    :param row_count: (int) total number of entries in original dataset
    :param targets (Set[str]) names of target features

    # Return
    :return: (List[FieldMetadata]) list of objects containing metadata for each field
    """

    metadata = []
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
                excluded=should_exclude(idx, field, dtype, row_count, targets),
                mode=infer_mode(field, targets),
                missing_values=missing_value_percent,
            )
        )

    # Count of number of initial non-text input features in the config, -1 for output
    input_count = sum(not meta.excluded and meta.mode == "input" and meta.config.type != TEXT for meta in metadata) - 1

    # Exclude text fields if no GPUs are available
    if resources["gpu"] == 0:
        for meta in metadata:
            if input_count > 2 and meta.config.type == TEXT:
                # By default, exclude text inputs when there are other candidate inputs
                meta.excluded = True

    return metadata


def infer_type(field: FieldInfo, missing_value_percent: float, row_count: int) -> str:
    """Perform type inference on field.

    # Inputs
    :param field: (FieldInfo) object describing field
    :param missing_value_percent: (float) percent of missing values in the column
    :param row_count: (int) total number of entries in original dataset

    # Return
    :return: (str) feature type
    """
    if field.dtype == DATE:
        return DATE

    num_distinct_values = field.num_distinct_values
    if num_distinct_values == 0:
        return CATEGORY
    distinct_values = field.distinct_values
    if num_distinct_values <= 2 and missing_value_percent == 0:
        # Check that all distinct values are conventional bools.
        if strings_utils.are_conventional_bools(distinct_values):
            return BINARY

    if field.image_values >= 3:
        return IMAGE

    if field.audio_values >= 3:
        return AUDIO

    # Use CATEGORY if:
    # - The number of distinct values is significantly less than the total number of examples.
    # - The distinct values are not all numbers.
    # - The distinct values are all numbers but comprise of a perfectly sequential list of integers that suggests the
    #   values represent categories.
    if num_distinct_values < row_count * CATEGORY_TYPE_DISTINCT_VALUE_PERCENTAGE_CUTOFF and (
        (not strings_utils.are_all_numbers(distinct_values)) or strings_utils.are_sequential_integers(distinct_values)
    ):
        return CATEGORY

    # Use NUMBER if all of the distinct values are numbers.
    if strings_utils.are_all_numbers(distinct_values):
        return NUMBER

    # TODO (ASN): add other modalities (image, etc. )
    # Fallback to TEXT.
    return TEXT


def should_exclude(idx: int, field: FieldInfo, dtype: str, row_count: int, targets: Set[str]) -> bool:
    if field.key == "PRI":
        return True

    if field.name in targets:
        return False

    if field.num_distinct_values == 0:
        return True

    distinct_value_percent = float(field.num_distinct_values) / row_count
    if distinct_value_percent == 1.0:
        upper_name = field.name.upper()
        if (idx == 0 and dtype == NUMBER) or upper_name.endswith("ID") or upper_name.startswith("ID"):
            return True

    return False


def infer_mode(field: FieldInfo, targets: Set[str] = None) -> str:
    if field.name in targets:
        return "output"
    if field.name.lower() == "split":
        return "split"
    return "input"
