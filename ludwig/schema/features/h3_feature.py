from marshmallow_dataclass import dataclass

from ludwig.constants import MISSING_VALUE_STRATEGY_OPTIONS, H3

from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BasePreprocessingConfig
from ludwig.schema.features.utils import register_preprocessor, PreprocessingDataclassField


@register_preprocessor(H3)
@dataclass
class H3PreprocessingConfig(BasePreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in an h3 column",
    )

    fill_value: int = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: int = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )


@dataclass
class H3InputFeatureConfig(BaseInputFeatureConfig):
    """H3InputFeatureConfig is a dataclass that configures the parameters used for an h3 input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=H3)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=H3,
        default="embed",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )
