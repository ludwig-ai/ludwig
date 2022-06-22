from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes
from ludwig.decoders.registry import get_decoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class TextInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='text'
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes('text').keys()),
        default="parallel_cnn",
        description="Encoder to use for this text feature.",
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class TextOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes('text').keys()),
        default="generator",
        description="Decoder to use for this text output feature.",
    )
