from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class NumberInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """NumberInputFeature is a dataclass that configures the parameters used for a number input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='number'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "dense"],
        default="passthrough",
        description="Encoder to use for this number feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class NumberOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):

    decoder: Optional[str] = schema_utils.StringOptions(
        ["regressor"],
        default="regressor",
        allow_none=True,
        description="Decoder to use for this number feature.",
    )

