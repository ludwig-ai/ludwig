import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union

from dataclasses_json import dataclass_json

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.misc_utils import memoized_method


@DeveloperAPI
class ExpectedImpact(int, Enum):
    """The expected impact of determining a "good" value for a specific parameter.

    - HIGH: this parameter should almost always be included in a hyperopt run and can make or break a good model.
    - MEDIUM: this parameter can sometimes make or break a good model.
    - LOW: this parameter usually does not have a significant impact on model performance.
    """

    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@DeveloperAPI
@dataclass_json()
@dataclass
class ParameterMetadata:
    """Contains descriptive information that pertains to a Ludwig configuration parameter."""

    # Quick description generally for UI display
    short_description: str = ""

    # In depth description generally for documentation purposes.
    long_description: str = ""

    # How this parameter can be displayed in a human-readable form.
    ui_display_name: Union[str, None] = ""

    # Why the default value for this parameter is the default.
    default_value_reasoning: Union[str, None] = None

    # Examples of other values that can be used for this parameter.
    example_value: Union[List[Any], None] = None

    # List of related parameters that this parameter interacts with or depends on.
    related_parameters: Union[List[str], None] = None

    # Other information that is relevant for this parameter.
    other_information: Union[str, None] = None

    # If we change/increase/decrease this parameter, what's the intuition for how model performance would change, i.e.:
    # learning curves, model speed, memory usage, etc.
    description_implications: Union[str, None] = None

    # What values would a machine learning expert suggest users try to help improve their model? Ideally, covers 95%
    # (~2 sigma) of use cases.
    suggested_values: Any = None

    # The reasoning behind the suggested values, as well as model performance indicators or other intuition that could
    # help inform a user to make an educated decision about what values to experiment with for this parameter.
    suggested_values_reasoning: Union[str, None] = None

    # True if this parameter could be frequently used, would have a high impact, and/or would be interesting for a
    # machine learning practitioner.
    commonly_used: bool = False

    # The expected impact of determining a "good" value for this parameter.
    expected_impact: ExpectedImpact = ExpectedImpact.UNKNOWN

    # List of links, papers, and blog posts to learn more.
    literature_references: Union[List[str], None] = None

    # Whether the parameter is used strictly internally.
    internal_only: bool = False

    @memoized_method(maxsize=1)
    def to_json_dict(self) -> Dict[str, Any]:
        return json.loads(self.to_json())


@DeveloperAPI
def convert_metadata_to_json(pm: ParameterMetadata) -> Dict[str, Any]:
    """Converts a ParameterMetadata dict to a normal JSON dict.

    NOTE: Without the json.loads call, to_json() returns
    a string repr that is improperly parsed.
    """
    if not pm:
        return None
    return pm.to_json_dict()


# This is a quick way to flag schema parameters as internal only via the `parameter_metadata` argument
INTERNAL_ONLY = ParameterMetadata(internal_only=True)
