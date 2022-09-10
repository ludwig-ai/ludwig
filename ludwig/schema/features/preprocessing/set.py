from marshmallow_dataclass import dataclass

from ludwig.constants import DROP_ROW, MISSING_VALUE_STRATEGY_OPTIONS, SET
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.utils import strings_utils


@register_preprocessor(SET)
@dataclass
class SetPreprocessingConfig(BasePreprocessingConfig):

    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
        "default value space splits the string on spaces. Common options include: underscore (splits on "
        "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
        "JSON parser).",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )


@register_preprocessor("set_output")
@dataclass
class SetOutputPreprocessingConfig(SetPreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set output feature",
    )
