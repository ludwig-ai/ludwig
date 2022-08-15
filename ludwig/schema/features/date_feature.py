from marshmallow_dataclass import dataclass

from ludwig.constants import MISSING_VALUE_STRATEGY_OPTIONS

from ludwig.constants import DATE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BasePreprocessingConfig
from ludwig.schema.features.utils import register_preprocessor, PreprocessingDataclassField


@register_preprocessor(DATE)
@dataclass
class DatePreprocessingConfig(BasePreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a date column",
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

    datetime_format: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="This parameter can either be a datetime format string, or null, in which case the datetime "
        "format will be inferred automatically.",
    )


@dataclass
class DateInputFeatureConfig(BaseInputFeatureConfig):
    """DateInputFeature is a dataclass that configures the parameters used for a date input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=DATE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=DATE,
        default="embed",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )
