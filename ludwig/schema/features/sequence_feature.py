from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import SEQUENCE
from ludwig.decoders.registry import get_decoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class SequenceInputFeatureConfig(BaseInputFeatureConfig):
    """SequenceInputFeatureConfig is a dataclass that configures the parameters used for a sequence input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SEQUENCE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=SEQUENCE,
        default='embed',
    )


@dataclass
class SequenceOutputFeatureConfig(BaseOutputFeatureConfig):
    """SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output
    feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(SEQUENCE).keys()),
        default="generator",
        allow_none=True,
        description="Decoder to use for this sequence feature.",
    )
