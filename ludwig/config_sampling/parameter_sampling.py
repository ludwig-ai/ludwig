import random
from typing import Any, Dict, List, Union

from ludwig.schema.metadata.parameter_metadata import ExpectedImpact

# base types for ludwig config parameters.
ParameterBaseTypes = Union[str, float, int, bool, None]


def handle_property_type(
    property_type: str, item: Dict[str, Any], expected_impact: ExpectedImpact = ExpectedImpact.HIGH
) -> List[Union[ParameterBaseTypes, List[ParameterBaseTypes]]]:
    """Return possible parameter values for a parameter type.

    Args:
        property_type: type of the parameter (e.g. array, number, etc.)
        item: dictionary containing details on the parameter such as default, min and max values.
        expected_impact: threshold expected impact that we'd like to include.
    """
    parameter_metadata = item.get("parameter_metadata", None)
    if not parameter_metadata:
        return []

    # don't explore internal only parameters.
    if parameter_metadata.get("internal_only", True):
        return []

    # don't explore parameters that have expected impact less than HIGH.
    if parameter_metadata.get("expected_impact", ExpectedImpact.LOW) < expected_impact:
        return []

    if property_type == "number":
        return explore_number(item)
    elif property_type == "integer":
        return explore_integer(item)
    elif property_type == "string":
        return explore_string(item)
    elif property_type == "boolean":
        return explore_boolean()
    elif property_type == "null":
        return explore_null()
    elif property_type == "array":
        return explore_array(item)
    else:
        return []


def explore_array(item: Dict[str, Any]) -> List[List[ParameterBaseTypes]]:
    """Return possible parameter values for the `array` parameter type.

    Args:
        item: dictionary containing details on the parameter such as default, min and max values.
    """

    candidates = []
    if "default" in item and item["default"]:
        candidates.append(item["default"])

    item_choices = []
    maxlen = 0

    # In the case where the length of the array isn't defined.
    if not isinstance(item["items"], list):
        return []

    for item_of in item["items"]:
        choices = handle_property_type(item_of["type"], item_of)
        maxlen = max(maxlen, len(choices))
        item_choices.append(choices)

    # pad to same length
    for i in range(len(item_choices)):
        item_choices[i] = maxlen * item_choices[i]
        item_choices[i] = item_choices[i][:maxlen]

    merged = list(zip(*item_choices)) + candidates
    return [list(tup) for tup in merged]


def explore_number(item: Dict[str, Any]) -> List[ParameterBaseTypes]:
    """Return possible parameter values for the `number` parameter type.

    Args:
        item: dictionary containing details on the parameter such as default, min and max values.
    TODO(Wael): Improve logic.
    """
    minimum, maximum = 0, 1
    if "default" not in item or item["default"] is None:
        candidates = []
    else:
        candidates = [1, 2, item["default"], 2 * (item["default"] + 1), item["default"] // 2, -1 * item["default"]]

    if "minimum" in item:
        minimum = item["minimum"]
        candidates = [num for num in candidates if num > minimum]
    if "maximum" in item:
        maximum = item["maximum"]
        candidates = [num for num in candidates if num < maximum]
    return candidates + [random.random() * 0.99 * maximum]


def explore_integer(item: Dict[str, Any]) -> List[ParameterBaseTypes]:
    """Return possible parameter values for the `integer` parameter type.

    Args:
        item: dictionary containing details on the parameter such as default, min and max values.
    TODO(Wael): Improve logic.
    """
    minimum, maximum = 0, 10

    if "default" not in item or item["default"] is None:
        candidates = []
    else:
        candidates = [item["default"], 2 * (item["default"] + 1), item["default"] // 2, -1 * item["default"]]

    if "minimum" in item:
        minimum = item["minimum"]
        candidates = [num for num in candidates if num >= item["minimum"]]
    if "maximum" in item:
        maximum = item["maximum"]
        candidates = [num for num in candidates if num <= item["maximum"]]

    return candidates + [random.randint(minimum, maximum)]


def explore_string(item: Dict[str, Any]) -> List[ParameterBaseTypes]:
    """Return possible parameter values for the `string` parameter type.

    Args:
        item: dictionary containing details on the parameter such as default, min and max values.
    """

    if "enum" in item:
        return item["enum"]
    return [item["default"]]


def explore_boolean() -> List[bool]:
    """Return possible parameter values for the `boolean` parameter type (i.e. [True, False])"""
    return [True, False]


def explore_null() -> List[None]:
    """Return possible parameter values for the `null` parameter type (i.e. [None])"""
    return [None]
