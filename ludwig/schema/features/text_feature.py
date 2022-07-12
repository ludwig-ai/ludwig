from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import TEXT
from ludwig.decoders.registry import get_decoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.encoders import BaseEncoderConfig, EncoderDataclassField


@dataclass
class TextInputFeatureConfig(BaseInputFeatureConfig):
    """TextInputFeatureConfig is a dataclass that configures the parameters used for a text input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TEXT)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=TEXT,
        default='parallel_cnn',
    )


@dataclass
class TextOutputFeatureConfig(BaseOutputFeatureConfig):
    """TextOutputFeatureConfig is a dataclass that configures the parameters used for a text output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(TEXT).keys()),
        default="generator",
        description="Decoder to use for this text output feature.",
    )
