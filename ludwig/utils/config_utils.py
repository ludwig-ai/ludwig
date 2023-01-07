from typing import Any, Dict, Set

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DECODER, ENCODER, IMAGE, INPUT_FEATURES, PREPROCESSING, SEQUENCE, TEXT, TIMESERIES, TYPE
from ludwig.encoders.registry import get_encoder_cls
from ludwig.features.feature_registries import get_input_type_registry
from ludwig.schema.model_config import get_default_decoder_type, get_default_encoder_type, ModelConfig
from ludwig.types import FeatureConfigDict, FeatureTypeDefaultsDict, PreprocessingConfigDict
from ludwig.utils.misc_utils import merge_dict


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


@DeveloperAPI
def merge_fixed_preprocessing_params(
    feature_type: str, preprocessing_params: Dict[str, Any], encoder_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Update preprocessing parameters if encoders require fixed preprocessing parameters."""
    encoder_type = encoder_params.get(TYPE, get_default_encoder_type(feature_type))
    encoder_class = get_encoder_cls(feature_type, encoder_type)
    return merge_dict(preprocessing_params, encoder_class.get_fixed_preprocessing_params(encoder_params))


@DeveloperAPI
def get_default_encoder_or_decoder(feature: FeatureConfigDict, config_feature_group: str) -> str:
    """Returns the default encoder or decoder for a feature."""
    if config_feature_group == INPUT_FEATURES:
        return get_default_decoder_type(feature[TYPE])
    else:
        return get_default_decoder_type(feature[TYPE])


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
    for feature in config.input_features.to_list():
        feature_type = feature.get("type")
        encoder = feature.get("encoder", {})

        encoder_type = encoder.get(TYPE, get_default_encoder_type(feature_type))
        encoder_class = get_encoder_cls(feature_type, encoder_type)
        if encoder_class.is_pretrained(encoder):
            return True

    return False
