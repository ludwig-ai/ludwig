import bisect
import logging
from dataclasses import dataclass
from typing import Dict, List

from dataclasses_json import dataclass_json, LetterCase
from pandas import Series

from ludwig.constants import COMBINER, CONFIG, HYPEROPT, NUMBER, PARAMETERS, SAMPLER, TRAINER, TYPE
from ludwig.utils.defaults import default_combiner_type

try:
    import ray
except ImportError:
    raise ImportError(" ray is not installed. " "In order to use auto_train please run " "pip install ludwig[ray]")


logger = logging.getLogger(__name__)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldInfo:
    name: str
    dtype: str
    key: str = None
    distinct_values: List = None
    num_distinct_values: int = 0
    nonnull_values: int = 0
    image_values: int = 0
    avg_words: int = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldConfig:
    name: str
    column: str
    type: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class FieldMetadata:
    name: str
    config: FieldConfig
    excluded: bool
    mode: str
    missing_values: float


def avg_num_tokens(field: Series) -> int:
    # sample a subset if dataframe is large
    if len(field) > 5000:
        field = field.sample(n=5000, random_state=40)
    unique_entries = field.unique()
    avg_words = round(Series(unique_entries).str.split().str.len().mean())
    return avg_words


def get_available_resources() -> dict:
    # returns total number of gpus and cpus
    resources = ray.cluster_resources()
    gpus = resources.get("GPU", 0)
    cpus = resources.get("CPU", 0)
    resources = {"gpu": gpus, "cpu": cpus}
    return resources


def get_model_name(config: dict) -> str:
    if COMBINER in config and TYPE in config[COMBINER]:
        return config[COMBINER][TYPE]
    return default_combiner_type


def _ray_init():
    if ray.is_initialized():
        return

    try:
        ray.init("auto", ignore_reinit_error=True)
    except ConnectionError:
        logger.info("Initializing new Ray cluster...")
        ray.init()


# ref_configs comes from a file storing the config for a high-performing model per reference dataset.
# If the automl model type matches that of any reference models, set the initial point_to_evaluate
# in the automl hyperparameter search to the config of the reference model with the closest-matching
# input numerical columns ratio.  This model config "transfer learning" can improve the automl search.
def _add_transfer_config(base_config: Dict, ref_configs: Dict) -> Dict:
    base_model_type = base_config[COMBINER][TYPE]
    base_model_numeric_ratio = _get_ratio_numeric_input_features(base_config["input_features"])
    min_numeric_ratio_distance = 1.0
    min_dataset = None

    for dataset in ref_configs["datasets"]:
        dataset_config = dataset[CONFIG]
        if base_model_type == dataset_config[COMBINER][TYPE]:
            dataset_numeric_ratio = _get_ratio_numeric_input_features(dataset_config["input_features"])
            ratio_distance = abs(base_model_numeric_ratio - dataset_numeric_ratio)
            if ratio_distance <= min_numeric_ratio_distance:
                min_numeric_ratio_distance = ratio_distance
                min_dataset = dataset

    if min_dataset is not None:
        logger.info("Transfer config from dataset {}".format(min_dataset["name"]))
        min_dataset_config = min_dataset[CONFIG]
        hyperopt_params = base_config[HYPEROPT][PARAMETERS]
        point_to_evaluate = {}
        _add_option_to_evaluate(point_to_evaluate, min_dataset_config, hyperopt_params, COMBINER)
        _add_option_to_evaluate(point_to_evaluate, min_dataset_config, hyperopt_params, TRAINER)
        base_config[HYPEROPT][SAMPLER]["search_alg"]["points_to_evaluate"] = [point_to_evaluate]
    return base_config


def _get_ratio_numeric_input_features(input_features: Dict) -> float:
    num_input_features = len(input_features)
    num_numeric_input = 0
    for input_feature in input_features:
        if input_feature[TYPE] == NUMBER:
            num_numeric_input = num_numeric_input + 1
    return num_numeric_input / num_input_features


# Update point_to_evaluate w/option value from dataset_config for options in hyperopt_params.
# Also, add option value to associated categories list if it is not already included.
def _add_option_to_evaluate(
    point_to_evaluate: Dict, dataset_config: Dict, hyperopt_params: Dict, option_type: str
) -> Dict:
    options = dataset_config[option_type]
    for option in options.keys():
        option_param = option_type + "." + option
        if option_param in hyperopt_params.keys():
            option_val = options[option]
            point_to_evaluate[option_param] = option_val
            if option_val not in hyperopt_params[option_param]["categories"]:
                bisect.insort(hyperopt_params[option_param]["categories"], option_val)
    return point_to_evaluate
