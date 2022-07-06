from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import IMAGE
from ludwig.encoders.registry import get_encoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class ImageInputFeatureConfig(BaseInputFeatureConfig):
    """ImageInputFeatureConfig is a dataclass that configures the parameters used for an image input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=IMAGE)

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(IMAGE).keys()),
        default="stacked_cnn",
        description="Encoder to use for this image feature.",
    )
