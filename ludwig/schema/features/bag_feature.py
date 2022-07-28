from marshmallow_dataclass import dataclass

from ludwig.constants import BAG
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class BagInputFeatureConfig(BaseInputFeatureConfig):
    """BagInputFeatureConfig is a dataclass that configures the parameters used for a bag input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BAG)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=BAG,
        default="embed",
    )
