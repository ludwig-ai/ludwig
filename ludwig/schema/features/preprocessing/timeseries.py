from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DROP_ROW, FILL_WITH_CONST, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING, TIMESERIES
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.tokenizers import tokenizer_registry


@ludwig_dataclass
class BaseTimeseriesPreprocessingConfig(BasePreprocessingConfig):
    tokenizer: str = schema_utils.StringOptions(
        tokenizer_registry.keys(),
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["tokenizer"],
    )

    timeseries_length_limit: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="Defines the maximum length of the timeseries. All timeseries longer than this limit are cut off.",
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["timeseries_length_limit"],
    )

    padding_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="Float value that is used for padding and replacing missing values within a row.",
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["padding_value"],
    )

    padding: str = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="The direction of the padding.",
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["padding"],
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description=(
            "What strategy to follow when there's a missing value in a column. Currently applies only to a row missing "
            "in its entirety, not invididual elements within the row. For now, `NaN` values within a row are filled "
            "using the `padding_value`."
        ),
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description=(
            "The value to replace missing values with in case the `missing_value_strategy` is `fill_with_const`."
        ),
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description=(
            "The internally computed fill value to replace missing values with in case the "
            "`missing_value_strategy` is `fill_with_mode` or `fill_with_mean`."
        ),
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["computed_fill_value"],
    )


@DeveloperAPI
@register_preprocessor(TIMESERIES)
@ludwig_dataclass
class TimeseriesPreprocessingConfig(BaseTimeseriesPreprocessingConfig):
    window_size: int = schema_utils.NonNegativeInteger(
        default=0,
        allow_none=False,
        description=(
            "Optional lookback window size used to convert a column-major dataset (one observation per row) "
            "into a row-major dataset (each row has a timeseries window of observations). Starting from a given "
            "observation, a sliding window is taken going `window_size - 1` rows back to form the timeseries input "
            "feature. If this value is left as 0, then it is assumed that the dataset has been provided in row-major "
            "format (i.e., it has already been preprocessed such that each row is a timeseries window)."
        ),
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when a row of data is missing.",
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["missing_value_strategy"],
    )


@DeveloperAPI
@register_preprocessor("timeseries_output")
@ludwig_dataclass
class TimeseriesOutputPreprocessingConfig(BaseTimeseriesPreprocessingConfig):
    horizon: int = schema_utils.NonNegativeInteger(
        default=0,
        allow_none=False,
        description=(
            "Optional forecasting horizon used to convert a column-major dataset (one observation per row) "
            "into a row-major dataset (each row has a timeseries window of observations). Starting from a given "
            "observation, a sliding window is token going `horizon` rows forward in time, excluding the observation "
            "in the current row. If this value is left as 0, then it is assumed that the dataset has been provided in "
            "row-major format (i.e., it has already been preprocessed such that each row is a timeseries window)."
        ),
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when a row of data is missing.",
        parameter_metadata=FEATURE_METADATA[TIMESERIES][PREPROCESSING]["missing_value_strategy"],
    )
