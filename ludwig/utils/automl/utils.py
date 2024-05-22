import bisect
import logging
from typing import Dict

from numpy import nan_to_num
from pandas import Series

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    BINARY,
    CATEGORY,
    COMBINER,
    CONFIG,
    HYPEROPT,
    IMBALANCE_DETECTION_RATIO,
    NAME,
    NUMBER,
    PARAMETERS,
    SEARCH_ALG,
    TRAINER,
    TYPE,
)
from ludwig.features.feature_registries import get_output_type_registry
from ludwig.modules.metric_registry import get_metric_objective
from ludwig.schema.combiners.utils import get_combiner_jsonschema

logger = logging.getLogger(__name__)


@DeveloperAPI
def avg_num_tokens_decoder(x):
    if x is None:
        return None
    if type(x) is bytes:
        return x.decode("utf-8")
    return str(x)


@DeveloperAPI
def avg_num_tokens(field: Series) -> int:
    logger.info(f"Calculating average number tokens for field {field.name} using sample of 100 rows.")
    field_sample = field.head(100).apply(avg_num_tokens_decoder)

    unique_entries = field_sample.unique()
    avg_words = round(nan_to_num(Series(unique_entries).str.split().str.len().mean()))
    return avg_words


@DeveloperAPI
def get_model_type(config: dict) -> str:
    if (
        "input_features" in config
        and len(config["input_features"]) == 1
        and "type" in config["input_features"][0]
        and config["input_features"][0]["type"] == "text"
    ):
        model_type = "text"
    elif COMBINER in config and TYPE in config[COMBINER]:
        model_type = config[COMBINER][TYPE]
    else:
        default_combiner_type = get_combiner_jsonschema()["properties"]["type"]["default"]
        model_type = default_combiner_type
    return model_type


# ref_configs comes from a file storing the config for a high-performing model per reference dataset.
# If the automl model type matches that of any reference models, set the initial point_to_evaluate
# in the automl hyperparameter search to the config of the reference model with the closest-matching
# input number columns ratio.  This model config "transfer learning" can improve the automl search.
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
        base_config[HYPEROPT][SEARCH_ALG]["points_to_evaluate"] = [point_to_evaluate]
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


@DeveloperAPI
def set_output_feature_metric(base_config):
    """If single output feature, set trainer and hyperopt metric and goal for that feature if not set."""
    if len(base_config["output_features"]) != 1:
        # If multiple output features, ludwig uses the goal of minimizing combined loss;
        # this could be revisited/refined in the future.
        return base_config
    output_name = base_config["output_features"][0][NAME]
    output_type = base_config["output_features"][0][TYPE]
    output_metric = get_output_type_registry()[output_type].get_schema_cls().default_validation_metric
    output_goal = get_metric_objective(output_metric)
    if "validation_field" not in base_config[TRAINER] and "validation_metric" not in base_config[TRAINER]:
        base_config[TRAINER]["validation_field"] = output_name
        base_config[TRAINER]["validation_metric"] = output_metric
    if (
        "output_feature" not in base_config[HYPEROPT]
        and "metric" not in base_config[HYPEROPT]
        and "goal" not in base_config[HYPEROPT]
    ):
        base_config[HYPEROPT]["output_feature"] = output_name
        base_config[HYPEROPT]["metric"] = output_metric
        base_config[HYPEROPT]["goal"] = output_goal
    return base_config


@DeveloperAPI
def has_imbalanced_output(base_config, features_metadata) -> bool:
    """Check binary and category output feature(s) for imbalance, i.e., low minority/majority instance count
    ratio."""
    imbalanced_output = False
    for output_feature in base_config["output_features"]:
        if output_feature[TYPE] == BINARY or output_feature[TYPE] == CATEGORY:
            for feature_metadata in features_metadata:
                if output_feature[NAME] == feature_metadata.name:
                    if feature_metadata.imbalance_ratio < IMBALANCE_DETECTION_RATIO:
                        logger.info(
                            f"Imbalance in {output_feature[NAME]}: minority/majority={feature_metadata.imbalance_ratio}"
                        )
                        imbalanced_output = True
                    break
    return imbalanced_output
