from typing import Optional
from ludwig.constants import AUDIO

from marshmallow_dataclass import dataclass

from ludwig.encoders.registry import get_encoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class AudioInputFeatureConfig(BaseInputFeatureConfig):
    """AudioFeatureInputFeature is a dataclass that configures the parameters used for an audio input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type=AUDIO
    )

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(AUDIO).keys()),
        default="parallel_cnn",
        description="Encoder to use for this audio feature.",
    )
