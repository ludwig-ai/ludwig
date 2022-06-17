from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.features import base
from ludwig.schema.features.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class VectorInputFeatureConfig(base.BaseFeatureConfig):
    """
    VectorInputFeatureConfig is a dataclass that configures the parameters used for a vector input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='vector'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        ["passthrough", "dense"],
        default="dense",
        description="Encoder to use for this vector feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class VectorOutputFeatureConfig(base.BaseFeatureConfig):
    """
    VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        ["projector"],
        default="projector",
        description="Decoder to use for this vector feature.",
    )

