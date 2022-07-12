from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import AUDIO
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.encoders import BaseEncoderConfig, EncoderDataclassField


@dataclass
class AudioInputFeatureConfig(BaseInputFeatureConfig):
    """AudioFeatureInputFeature is a dataclass that configures the parameters used for an audio input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=AUDIO)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=AUDIO,
        default='parallel_cnn',
    )


