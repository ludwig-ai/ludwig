from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import BinaryPreprocessingConfig


@dataclass
class BinaryInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """BinaryInputFeature is a dataclass that configures the parameters used for a binary input feature."""

    preprocessing: Optional[str] = BinaryPreprocessingConfig(
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "dense"],
        default="passthrough",
        description="Encoder to use for this binary feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class BinaryOutputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """BinaryOutputFeature is a dataclass that configures the parameters used for a binary output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        ["regressor"],
        default="classifier",
        allow_none=True,
        description="Decoder to use for this binary feature.",
    )
