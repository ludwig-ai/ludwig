from marshmallow_dataclass import dataclass

from ludwig.constants import IMAGE
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class ImageInputFeatureConfig(BaseInputFeatureConfig):
    """ImageInputFeatureConfig is a dataclass that configures the parameters used for an image input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=IMAGE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=IMAGE,
        default="stacked_cnn",
    )
