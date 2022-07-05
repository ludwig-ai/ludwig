from typing import Optional
from ludwig.constants import SEQUENCE

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
        feature_type=SEQUENCE
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(SEQUENCE).keys()),
        default="embed",
        description="Encoder to use for this sequence feature.",
    )


@dataclass
class SequenceOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(SEQUENCE).keys()),
        default="generator",
        allow_none=True,
        description="Decoder to use for this sequence feature.",
    )
