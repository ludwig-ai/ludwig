from typing import Any, Dict, Set, Union

from ludwig.constants import TYPE


def get_feature_type_parameter_values_from_section(
    config: Dict[str, Any], features_section: str, feature_type: str, parameter_name: str
) -> Set:
    """Returns the set of all parameter values used for the given features_section, feature_type, and
    parameter_name."""
    parameter_values = set()
    for feature in config[features_section]:
        if feature[TYPE] == feature_type:
            parameter_values.add(feature[parameter_name])
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
