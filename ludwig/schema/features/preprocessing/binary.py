from typing import Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    BFILL,
    BINARY,
    DROP_ROW,
    FFILL,
    FILL_WITH_FALSE,
    FILL_WITH_MODE,
    FILL_WITH_TRUE,
    PREPROCESSING,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils import strings_utils


@DeveloperAPI
@register_preprocessor(BINARY)
@ludwig_dataclass
class BinaryPreprocessingConfig(BasePreprocessingConfig):
    """BinaryPreprocessingConfig is a dataclass that configures the parameters used for a binary input feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        [FILL_WITH_MODE, BFILL, FFILL, DROP_ROW, FILL_WITH_FALSE, FILL_WITH_TRUE],
        default=FILL_WITH_FALSE,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a binary column",
        parameter_metadata=FEATURE_METADATA[BINARY][PREPROCESSING]["missing_value_strategy"],
    )

    fallback_true_label: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The label to interpret as 1 (True) when the binary feature doesn't have a "
        "conventional boolean value",
        parameter_metadata=FEATURE_METADATA[BINARY][PREPROCESSING]["fallback_true_label"],
    )

    fill_value: Union[int, float, str] = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        field_options=[
            schema_utils.FloatRange(default=None, allow_none=True, min=0, max=1, description=""),
            schema_utils.StringOptions(options=strings_utils.all_bool_strs(), default="Y", allow_none=False),
            schema_utils.Boolean(default=True, description=""),
        ],
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[BINARY][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: Union[int, float, str] = schema_utils.OneOfOptionsField(
        default=None,
        allow_none=True,
        field_options=[
            schema_utils.FloatRange(default=1.0, allow_none=False, min=0, max=1, description=""),
            schema_utils.StringOptions(options=strings_utils.all_bool_strs(), default="Y", allow_none=False),
            schema_utils.Boolean(default=True, description=""),
        ],
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[BINARY][PREPROCESSING]["computed_fill_value"],
    )


@DeveloperAPI
@register_preprocessor("binary_output")
@ludwig_dataclass
class BinaryOutputPreprocessingConfig(BinaryPreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        [FILL_WITH_MODE, BFILL, FFILL, DROP_ROW, FILL_WITH_FALSE, FILL_WITH_TRUE],
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in a binary output feature",
        parameter_metadata=FEATURE_METADATA[BINARY][PREPROCESSING]["missing_value_strategy"],
    )

    fallback_true_label: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The label to interpret as 1 (True) when the binary feature doesn't have a "
        "conventional boolean value",
        parameter_metadata=FEATURE_METADATA[BINARY][PREPROCESSING]["fallback_true_label"],
    )
