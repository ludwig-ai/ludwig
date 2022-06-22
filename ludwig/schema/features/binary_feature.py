from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes
from ludwig.decoders.registry import get_decoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class BinaryInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """BinaryInputFeature is a dataclass that configures the parameters used for a binary input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='binary'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes('binary').keys()),
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
class BinaryOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """BinaryOutputFeature is a dataclass that configures the parameters used for a binary output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes('binary').keys()),
        default="regressor",
        allow_none=True,
        description="Decoder to use for this binary feature.",
    )
