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
        description=(
            "Normalization strategy to use for this number feature. If the value is `null` no normalization is "
            "performed."
        ),
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["normalization"],
    )

    outlier_strategy: Optional[str] = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS + [FILL_WITH_MEAN, None],
        default=None,
        allow_none=True,
        description=(
            "Determines how outliers will be handled in the dataset. In most cases, replacing outliers with the "
            "column mean (`fill_with_mean`) will be sufficient, but in others the outliers may be damaging enough "
            "to merit dropping the entire row of data (`drop_row`). In some cases, the best way to handle outliers "
            "is to leave them in the data, which is the behavior when this parameter is left as `null`."
        ),
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["outlier_strategy"],
    )

    outlier_threshold: Optional[float] = schema_utils.FloatRange(
        default=3.0,
        allow_none=False,
        min=0.0,
        description=(
            "Standard deviations from the mean past which a value is considered an outlier. The 3-sigma "
            "rule in statistics tells us that when data is normally distributed, 95% of the data will lie within 2 "
            "standard deviations of the mean, and greater than 99% of the data will lie within 3 standard deviations "
            "of the mean (see: [68–95–99.7 rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)). "
            "As such anything farther away than that is highly likely to be an outlier, and may distort the learning "
            "process by disproportionately affecting the model."
        ),
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["outlier_threshold"],
    )

    computed_outlier_fill_value: float = schema_utils.FloatRange(
        default=0.0,
        allow_none=False,
        description="The internally computed fill value to replace outliers with in case the "
        "outlier_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["computed_outlier_fill_value"],
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
