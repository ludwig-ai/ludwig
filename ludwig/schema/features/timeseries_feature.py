from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import TIMESERIES
from ludwig.encoders.registry import get_encoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField


@dataclass
class TimeseriesInputFeatureConfig(BaseInputFeatureConfig):
    """TimeseriesInputFeatureConfig is a dataclass that configures the parameters used for a timeseries input
    feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TIMESERIES)

    encoder: Optional[str] = schema_utils.StringOptions(
        list(get_encoder_classes(TIMESERIES).keys()),
        default="parallel_cnn",
        description="Encoder to use for this timeseries feature.",
    )
