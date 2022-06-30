from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union


class ExpectedImpact(Enum):
    """The expected impact of determining a "good" value for a specific parameter.

    - HIGH: this parameter should almost always be included in a hyperopt run and can make or break a good model.
    - MEDIUM: this parameter might make or break a good model.
    - LOW: this parmater usually does not have a significant impact on model performance.
    """

    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class ParameterMetadata:
    """ParameterMetadata is a dataclass that describes metadata fields for any Ludwig parameter."""

    # How the parameter can be displayed in a human-readable form.
    ui_display_name: str = ""

    # Why the default value is the default.
    default_value_reasoning: Union[str, None] = None

    # Examples of other values that can be used.
    example_value: List[Any] = None

    # List of related parameters that this parameter interacts with or depends on.
    related_parameters: Union[List[str], None] = None

    # Other information that is relevant for this parameter.
    other_information: Union[str, None] = None

    # If we change/increase/decrease this parameter, what's the intuition for how model performance would change, i.e.:
    # learning curves, model speed, memory usage, etc.
    description_implications: Union[str, None] = None

    # What values would you suggest users try? Ideally, covers 95% (~2 sigma) of use cases.
    suggested_values: Any = None

    # Indicators that would inform a user to try to use a suggested value, and why.
    suggested_values_reasoning: Union[str, None] = None

    # True if you believe changing this parameter could be frequently used, would have a high impact, and/or would be
    # interesting for a machine learning practitioner.
    commonly_used: bool = False

    # What's the expected impact of determining a "good" value for this parameter?
    #   HIGH: this parameter should almost always be included in a hyperopt run and can make or break a good model.
    #   MEDIUM: this parameter might make or break a good model.
    #   LOW: this parmater usually does not have a significant impact on model performance.
    expected_impact: ExpectedImpact = ExpectedImpact.UNKNOWN

    # List of links, papers, and blog posts to learn more.
    literature_references: Union[List[str], None] = None

    # Whether the parameter is used strictly internally.
    internal_only: bool = False
