from typing import Any, Dict, Set, Union

from ludwig.constants import DECODER, ENCODER, INPUT_FEATURES, PREPROCESSING, TYPE
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.utils.misc_utils import get_from_registry


def get_feature_type_parameter_values_from_section(
    config: Dict[str, Any], features_section: str, feature_type: str, parameter_name: str
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


def get_defaults_section_for_feature_type(
    feature_type: str,
    config_defaults: Dict[str, Dict[str, Any]],
    config_defaults_section: str,
) -> Union[Dict[str, Any], Dict]:
    """Returns a dictionary of all default parameter values specified in the global defaults section for the
    config_defaults_section of the feature_type."""

    if feature_type not in config_defaults:
        return {}

    if config_defaults_section not in config_defaults[feature_type]:
        return {}

    return config_defaults[feature_type][config_defaults_section]


def merge_config_preprocessing_with_feature_specific_defaults(
    config_preprocessing: Dict[str, Any], config_defaults: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Returns a new dictionary that merges preprocessing section of config with type-specific preprocessing
    parameters from config defaults."""
    preprocessing_params = {}
    preprocessing_params.update(config_preprocessing)
    for feature_type in config_defaults:
        preprocessing_params[feature_type] = config_defaults[feature_type].get(PREPROCESSING, {})
    return preprocessing_params


def get_default_encoder_or_decoder(feature: Dict[str, Any], config_feature_group: str) -> str:
    """Returns the default encoder or decoder for a feature."""
    if config_feature_group == INPUT_FEATURES:
        feature_schema = get_from_registry(feature.get(TYPE), input_type_registry).get_schema_cls()
        return feature_schema().encoder.type
    feature_schema = get_from_registry(feature.get(TYPE), output_type_registry).get_schema_cls()
    return feature_schema().decoder.type
