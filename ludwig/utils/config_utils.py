from typing import Any, Dict, Set

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
