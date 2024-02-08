from typing import Any, Dict, List, Set, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    DECODER,
    ENCODER,
    IMAGE,
    INPUT_FEATURES,
    MODEL_ECD,
    MODEL_LLM,
    MODEL_TYPE,
    PREPROCESSING,
    SEQUENCE,
    TEXT,
    TIMESERIES,
    TYPE,
)
from ludwig.features.feature_registries import get_input_type_registry
from ludwig.schema.model_config import ModelConfig
from ludwig.types import FeatureConfigDict, FeatureTypeDefaultsDict, PreprocessingConfigDict


@DeveloperAPI
def get_feature_type_parameter_values_from_section(
    config: ModelConfig, features_section: str, feature_type: str, parameter_name: str
) -> Set:
    """Returns the set of all parameter values used for the given features_section, feature_type, and
    parameter_name."""
    parameter_values = set()
    for feature in config[features_section]:
        if feature[TYPE] == feature_type:
            if parameter_name in feature:
                parameter_values.add(feature[parameter_name])
            elif parameter_name in feature[ENCODER]:
                parameter_values.add(feature[ENCODER][parameter_name])
            elif parameter_name in feature[DECODER]:
                parameter_values.add(feature[DECODER][parameter_name])
    return parameter_values


@DeveloperAPI
def get_defaults_section_for_feature_type(
    feature_type: str,
    config_defaults: FeatureTypeDefaultsDict,
    config_defaults_section: str,
) -> FeatureConfigDict:
    """Returns a dictionary of all default parameter values specified in the global defaults section for the
    config_defaults_section of the feature_type."""

    if feature_type not in config_defaults:
        return {}

    if config_defaults_section not in config_defaults[feature_type]:
        return {}

    return config_defaults[feature_type][config_defaults_section]


def get_preprocessing_params(config_obj: ModelConfig) -> PreprocessingConfigDict:
    """Returns a new dictionary that merges preprocessing section of config with type-specific preprocessing
    parameters from config defaults."""
    preprocessing_params = {}
    preprocessing_params.update(config_obj.preprocessing.to_dict())
    for feat_type in get_input_type_registry().keys():
        if hasattr(config_obj.defaults, feat_type):
            preprocessing_params[feat_type] = getattr(config_obj.defaults, feat_type).preprocessing.to_dict()
    return preprocessing_params


@DeveloperAPI
def merge_config_preprocessing_with_feature_specific_defaults(
    config_preprocessing: PreprocessingConfigDict, config_defaults: FeatureTypeDefaultsDict
) -> PreprocessingConfigDict:
    """Returns a new dictionary that merges preprocessing section of config with type-specific preprocessing
    parameters from config defaults."""
    preprocessing_params = {}
    preprocessing_params.update(config_preprocessing)
    for feature_type in config_defaults:
        preprocessing_params[feature_type] = config_defaults[feature_type].get(PREPROCESSING, {})
    return preprocessing_params


def has_trainable_encoder(config: ModelConfig) -> bool:
    for feature in config.input_features.to_list():
        encoder = feature.get("encoder", {})
        if encoder.get("trainable", False):
            # TODO(travis): we assume here that False is always the default, which may not be true. We should dervice
            # this from the schema.
            return True

    return False


def has_unstructured_input_feature(config: ModelConfig) -> bool:
    for feature in config.input_features.to_list():
        if feature.get("type", None) in {TEXT, IMAGE, SEQUENCE, TIMESERIES}:
            return True
    return False


def has_pretrained_encoder(config: ModelConfig) -> bool:
    for feature in config.input_features:
        if feature.encoder.is_pretrained():
            return True
    return False


def config_uses_llm(config: Union[Dict[str, Any], ModelConfig]) -> bool:
    """Determine if a config uses an LLM.

    Args:
        config: Ludwig config object or dictionary

    Returns:
        True if the model type is LLM or if the model uses and LLM encoder, otherwise False.
    """
    uses_llm = False

    # For a valid config, model_type LLM is automatically True
    # ECD or GBM models need to be checked for at least one LLM text encoder
    if isinstance(config, ModelConfig):
        if config.model_type == MODEL_LLM:
            uses_llm = True
        else:
            for feature in config.input_features:
                if feature.encoder and feature.encoder.type == MODEL_LLM:
                    uses_llm = True
                    break
    elif isinstance(config, dict) and config:
        if config.get(MODEL_TYPE, MODEL_ECD) == MODEL_LLM:
            uses_llm = True
        elif INPUT_FEATURES in config:
            for feature in config.get(INPUT_FEATURES, []):
                if feature.get(ENCODER, {}).get(TYPE) == MODEL_LLM:
                    uses_llm = True
                    break
        else:
            raise ValueError(
                "Invalid config cannot be checked for LLM usage because it has no input features." f"Config: {config}"
            )
    else:
        raise ValueError(f"Invalid config cannot be checked for LLM usage. Config: {config}")

    return uses_llm


def get_quantization(config: Union[Dict[str, Any], ModelConfig]) -> Union[List[int], None]:
    """Get the quantization specified in a config at any level.

    Args:
        config: Ludwig config object or dictionary

    Returns:
        For LLM models, the value of quantization.bits or None if it is not specified.
        For ECD and GBM models, the list of values of quantization.bits for each encoder. If the encoder does not
        support quantization or no quantization config is specified, the list entry is None.
    """
    if isinstance(config, ModelConfig):
        if config.model_type == MODEL_LLM:
            return [config.quantization.bits] if config.quantization else None
        else:
            quantization_bits = []
            for feature in config.input_features:
                try:
                    quantization = feature.encoder.quantization.bits
                except AttributeError:
                    quantization = None
                quantization_bits.append(quantization)
            return quantization_bits
    elif isinstance(config, dict) and config:
        if config.get(MODEL_TYPE, MODEL_ECD) == MODEL_LLM:
            quantization = config.get("quantization", {}).get("bits")
            return [quantization] if quantization is not None else quantization_bits
        elif INPUT_FEATURES in config:
            quantization_bits = []
            for feature in config.get(INPUT_FEATURES, []):
                quantization_bits.append(feature.get(ENCODER, {}).get("quantization", {}).get("bits"))
            return quantization_bits
        else:
            raise ValueError(
                "Invalid config cannot be checked for quantization because it has no input features."
                f"Config: {config}"
            )
    else:
        raise ValueError(f"Invalid config cannot be checked for quantization. Config: {config}")
