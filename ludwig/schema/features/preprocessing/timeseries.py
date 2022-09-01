from marshmallow_dataclass import dataclass

from ludwig.constants import MISSING_VALUE_STRATEGY_OPTIONS, TIMESERIES
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.utils.tokenizers import tokenizer_registry


@register_preprocessor(TIMESERIES)
@dataclass
class TimeseriesPreprocessingConfig(BasePreprocessingConfig):

    tokenizer: str = schema_utils.StringOptions(
        sorted(list(tokenizer_registry.keys())),
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    timeseries_length_limit: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="Defines the maximum length of the timeseries. All timeseries longer than this limit are cut off.",
    )

    padding_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="Float value that is used for padding.",
    )

    padding: str = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )
