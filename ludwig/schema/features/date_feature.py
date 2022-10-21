from marshmallow_dataclass import dataclass

from ludwig.constants import DATE
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import input_config_registry, input_mixin_registry
from ludwig.schema.schema_utils import BaseMarshmallowConfig


@input_mixin_registry.register(DATE)
@dataclass
class DateInputFeatureConfigMixin(BaseMarshmallowConfig):
    """DateInputFeatureConfigMixin is a dataclass that configures the parameters used in both the date input
    feature and the date global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=DATE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=DATE,
        default="embed",
    )


@input_config_registry.register(DATE)
@dataclass(repr=False)
class DateInputFeatureConfig(BaseInputFeatureConfig, DateInputFeatureConfigMixin):
    """DateInputFeature is a dataclass that configures the parameters used for a date input feature."""

    pass
