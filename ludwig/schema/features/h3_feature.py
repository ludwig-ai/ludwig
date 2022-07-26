from marshmallow_dataclass import dataclass

from ludwig.constants import H3
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class H3InputFeatureConfig(BaseInputFeatureConfig):
    """H3InputFeatureConfig is a dataclass that configures the parameters used for an h3 input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=H3)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=H3,
        default="embed",
    )
