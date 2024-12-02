from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DATE, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import ecd_defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@ecd_defaults_config_registry.register(DATE)
@input_mixin_registry.register(DATE)
@ludwig_dataclass
class DateInputFeatureConfigMixin(BaseMarshmallowConfig):
    """DateInputFeatureConfigMixin is a dataclass that configures the parameters used in both the date input
    feature and the date global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=DATE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=DATE,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(DATE)
@ludwig_dataclass
class DateInputFeatureConfig(DateInputFeatureConfigMixin, BaseInputFeatureConfig):
    """DateInputFeature is a dataclass that configures the parameters used for a date input feature."""

    type: str = schema_utils.ProtectedString(DATE)
