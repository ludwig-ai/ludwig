from marshmallow_dataclass import dataclass

from ludwig.constants import TIMESERIES
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class TimeseriesInputFeatureConfig(BaseInputFeatureConfig):
    """TimeseriesInputFeatureConfig is a dataclass that configures the parameters used for a timeseries input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TIMESERIES)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=TIMESERIES,
        default="parallel_cnn",
    )
