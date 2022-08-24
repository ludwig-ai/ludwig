from marshmallow_dataclass import dataclass

from ludwig.constants import AUDIO
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class AudioInputFeatureConfig(BaseInputFeatureConfig):
    """AudioFeatureInputFeature is a dataclass that configures the parameters used for an audio input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=AUDIO)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=AUDIO,
        default="parallel_cnn",
    )

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )
