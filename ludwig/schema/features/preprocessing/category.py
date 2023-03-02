from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, DROP_ROW, FILL_WITH_CONST, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils import strings_utils


@DeveloperAPI
@register_preprocessor(CATEGORY)
@ludwig_dataclass
class CategoryPreprocessingConfig(BasePreprocessingConfig):
    """CategoryPreprocessingConfig is a dataclass that configures the parameters used for a category input
    feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category column",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description=(
            "The value to replace missing values with in case the `missing_value_strategy` is `fill_with_const`"
        ),
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["computed_fill_value"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["most_common"],
    )


@DeveloperAPI
@register_preprocessor("category_output")
@ludwig_dataclass
class CategoryOutputPreprocessingConfig(CategoryPreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category output feature",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["missing_value_strategy"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[CATEGORY][PREPROCESSING]["most_common"],
    )
