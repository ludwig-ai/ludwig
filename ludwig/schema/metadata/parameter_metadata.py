import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
class ComputeTier(int, Enum):
    """The compute tier defines the type of compute resources that a model typically needs to get good
    throughput."""

    CPU = 0
    """Model can train effectively on CPU hardware."""

    GPU_LOW = 1
    """Model can train effectively on commodity GPU hardware, or inference optimized SKUs like NVIDIA T4."""

    GPU_MEDIUM = 2
    """Model can train effectively on training-optimized GPU hardware like V100, A10G, or A5000."""

    GPU_HIGH = 3
    """Model requires high-end GPUs like A100 or H100 to achieve good throughput."""


@DeveloperAPI
@dataclass_json()
@dataclass
class ParameterMetadata:
    """Contains descriptive information that pertains to a Ludwig configuration parameter."""

    short_description: str = ""
    """Quick description generally for UI display."""

    long_description: str = ""
    """In depth description generally for documentation purposes."""

    ui_display_name: Union[str, None] = ""
    """How this parameter can be displayed in a human-readable form."""

    default_value_reasoning: Union[str, None] = None
    """The reasoning behind the default value for this parameter."""

    example_value: Union[List[Any], None] = None
    """Examples of other values that can be used for this parameter."""

    related_parameters: Union[List[str], None] = None
    """List of related parameters that this parameter interacts with or depends on."""

    other_information: Union[str, None] = None
    """Other information that is relevant for this parameter."""

    description_implications: Union[str, None] = None
    """The intuition for how model performance would change if this parameter is changed."""

    suggested_values: Any = None
    """What values would a machine learning expert suggest users try to help improve their model?

    Should cover 95% (2-sigma) worth of use-cases.
    """

    suggested_values_reasoning: Union[str, None] = None
    """The reasoning behind the suggested values, as well as model performance indicators or other intuition that
    could help inform a user to make an educated decision about what values to experiment with for this
    parameter."""

    commonly_used: bool = False
    """True if this parameter could be frequently used, would have a high impact, and/or would be interesting for a
    machine learning practitioner."""

    expected_impact: ExpectedImpact = ExpectedImpact.UNKNOWN
    """The expected impact of determining a "good" value for this parameter."""

    literature_references: Union[List[str], None] = None
    """List of links, papers, and blog posts to learn more."""

    internal_only: bool = False
    """True if this parameter is used strictly internally and should not be exposed to users."""

    compute_tier: ComputeTier = ComputeTier.CPU
    """The compute tier defines the type of compute resources that a model typically needs to get good
    throughput."""

    ui_component_type: Optional[str] = None
    """Override for HTML component type that should be used to render this field in UIs."""

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
        return ParameterMetadata().to_json_dict()
    return pm.to_json_dict()


# This is a quick way to flag schema parameters as internal only via the `parameter_metadata` argument
INTERNAL_ONLY = ParameterMetadata(internal_only=True)
