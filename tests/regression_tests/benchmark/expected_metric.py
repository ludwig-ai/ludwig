from dataclasses import dataclass
from enum import Enum
from typing import Union

from dataclasses_json import dataclass_json


class MetricRegressionDirection(Enum):
    """Which direction is considered a regression."""

    LOWER = -1
    HIGHER = 1


@dataclass_json
@dataclass
class ExpectedMetric:
    # Output feature name.
    output_feature_name: str

    # Metric name.
    metric_name: str

    # Expected metric value.
    expected_value: Union[int, float]

    # Which direction is considered a regression.
    regression_direction: int

    # The percentage change that would trigger a notification/failure.
    percent_change_sensitivity: float

    def __post_init__(self):
        if self.regression_direction == "LOWER":
            self.regression_direction = -1
        elif self.regression_direction == "HIGHER":
            self.regression_direction = 1
        else:
            raise ValueError(
                "Regression direction in the expected performance YAML file should be one" "of 'LOWER', 'HIGHER'."
            )
