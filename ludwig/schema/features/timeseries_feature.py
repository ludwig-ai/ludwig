from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TIMESERIES
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig


@DeveloperAPI
@input_mixin_registry.register(TIMESERIES)
@dataclass
class TimeseriesInputFeatureConfigMixin(BaseMarshmallowConfig):
    """TimeseriesInputFeatureConfigMixin is a dataclass that configures the parameters used in both the timeseries
    input feature and the timeseries global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TIMESERIES)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=TIMESERIES,
        default="parallel_cnn",
    )


@DeveloperAPI
@input_config_registry.register(TIMESERIES)
@dataclass(repr=False, order=True)
class TimeseriesInputFeatureConfig(BaseInputFeatureConfig, TimeseriesInputFeatureConfigMixin):
    """TimeseriesInputFeatureConfig is a dataclass that configures the parameters used for a timeseries input
    feature."""

    pass
