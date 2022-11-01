from marshmallow_dataclass import dataclass

from ludwig.constants import DROP_ROW, MISSING_VALUE_STRATEGY_OPTIONS, NUMBER, PREPROCESSING
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata.feature_metadata import FEATURE_METADATA


@register_preprocessor(NUMBER)
@dataclass(repr=False)
class NumberPreprocessingConfig(BasePreprocessingConfig):
    """NumberPreprocessingConfig is a dataclass that configures the parameters used for a number input feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
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
        ["zscore", "minmax", "log1p"],
        default=None,
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["normalization"],
    )


@register_preprocessor("number_output")
@dataclass(repr=False)
class NumberOutputPreprocessingConfig(NumberPreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number output feature",
        parameter_metadata=FEATURE_METADATA[NUMBER][PREPROCESSING]["missing_value_strategy"],
    )
