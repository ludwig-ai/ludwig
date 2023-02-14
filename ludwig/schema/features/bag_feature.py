from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BAG, MODEL_ECD
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@defaults_config_registry.register(BAG)
@input_mixin_registry.register(BAG)
@ludwig_dataclass
class BagInputFeatureConfigMixin(BaseMarshmallowConfig):
    """BagInputFeatureConfigMixin is a dataclass that configures the parameters used in both the bag input feature
    and the bag global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BAG)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=BAG,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(BAG)
@ludwig_dataclass
class BagInputFeatureConfig(BaseInputFeatureConfig, BagInputFeatureConfigMixin):
    """BagInputFeatureConfig is a dataclass that configures the parameters used for a bag input feature."""

    pass
