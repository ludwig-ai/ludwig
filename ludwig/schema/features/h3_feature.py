from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import H3, MODEL_ECD
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@defaults_config_registry.register(H3)
@input_mixin_registry.register(H3)
@ludwig_dataclass
class H3InputFeatureConfigMixin(BaseMarshmallowConfig):
    """H3InputFeatureConfigMixin is a dataclass that configures the parameters used in both the h3 input feature
    and the h3 global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=H3)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=H3,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(H3)
@ludwig_dataclass
class H3InputFeatureConfig(BaseInputFeatureConfig, H3InputFeatureConfigMixin):
    """H3InputFeatureConfig is a dataclass that configures the parameters used for an h3 input feature."""

    pass
