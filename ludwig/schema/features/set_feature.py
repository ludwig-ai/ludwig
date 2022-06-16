from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import SetPreprocessingConfig


@dataclass
class SetInputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """
    SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature.
    """

    preprocessing: Optional[str] = SetPreprocessingConfig(
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["embed"],
        default="embed",
        description="Encoder to use for this set feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class SetOutputFeatureConfig(schema_utils.BaseMarshmallowConfig, base.BaseFeatureConfig):
    """
    SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        ["classifier"],
        default="classifier",
        allow_none=True,
        description="Decoder to use for this set feature.",
    )
