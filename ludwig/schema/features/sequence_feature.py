from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes
from ludwig.decoders.registry import get_decoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class SequenceInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    SequenceInputFeatureConfig is a dataclass that configures the parameters used for a sequence input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='sequence'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes('sequence').keys()),
        default="embed",
        description="Encoder to use for this sequence feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class SequenceOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes('sequence').keys()),
        default="generator",
        allow_none=True,
        description="Decoder to use for this sequence feature.",
    )
