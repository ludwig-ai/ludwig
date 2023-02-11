from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    DROP_ROW,
    FILL_WITH_CONST,
    FILL_WITH_MEAN,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NUMBER,
    PREPROCESSING,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_preprocessor(NUMBER)
@ludwig_dataclass
class NumberPreprocessingConfig(BasePreprocessingConfig):
    """NumberPreprocessingConfig is a dataclass that configures the parameters used for a number input feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS + [FILL_WITH_MEAN],
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number column",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: float = schema_utils.FloatRange(
        default=0.0,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: float = schema_utils.FloatRange(
        default=0.0,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["computed_fill_value"],
    )

    normalization: str = schema_utils.StringOptions(
        ["zscore", "minmax", "log1p", "iq"],
        default="zscore",
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["normalization"],
    )

    outlier_strategy: Optional[str] = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS + [FILL_WITH_MEAN, None],
        default=None,
        allow_none=True,
        description="What strategy to follow when there's an outlier in a number column, "
        "defaults to doing nothing (leaving the outliers as-is)",
    )

    outlier_threshold: Optional[float] = schema_utils.FloatRange(
        default=3.0,
        allow_none=False,
        min=0.0,
        description="Standard deviations from the mean past which a value is considered an outlier",
    )

    computed_outlier_fill_value: float = schema_utils.FloatRange(
        default=0.0,
        allow_none=False,
        description="The internally computed fill value to replace outliers with in case the "
        "outlier_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["computed_fill_value"],
    )


@DeveloperAPI
@register_preprocessor("number_output")
@ludwig_dataclass
class NumberOutputPreprocessingConfig(NumberPreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS + [FILL_WITH_MEAN],
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number output feature",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["missing_value_strategy"],
    )

    normalization: str = schema_utils.StringOptions(
        ["zscore", "minmax", "log1p", "iq"],
        default=None,
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["normalization"],
    )
