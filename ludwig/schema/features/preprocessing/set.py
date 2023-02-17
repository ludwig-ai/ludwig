from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DROP_ROW, FILL_WITH_CONST, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING, SET
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils import strings_utils


@DeveloperAPI
@register_preprocessor(SET)
@ludwig_dataclass
class SetPreprocessingConfig(BasePreprocessingConfig):
    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
        "default value space splits the string on spaces. Common options include: underscore (splits on "
        "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
        "JSON parser).",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["tokenizer"],
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["computed_fill_value"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["most_common"],
    )


@DeveloperAPI
@register_preprocessor("set_output")
@ludwig_dataclass
class SetOutputPreprocessingConfig(SetPreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set output feature",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["missing_value_strategy"],
    )

    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
        "default value space splits the string on spaces. Common options include: underscore (splits on "
        "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
        "JSON parser).",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["tokenizer"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["lowercase"],
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[SET][PREPROCESSING]["most_common"],
    )
