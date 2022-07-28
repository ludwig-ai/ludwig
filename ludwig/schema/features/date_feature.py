from marshmallow_dataclass import dataclass

from ludwig.constants import DATE
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class DateInputFeatureConfig(BaseInputFeatureConfig):
    """DateInputFeature is a dataclass that configures the parameters used for a date input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=DATE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=DATE,
        default="embed",
    )
