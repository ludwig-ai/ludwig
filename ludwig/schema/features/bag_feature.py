from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BAG
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import ecd_input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig


@DeveloperAPI
@input_mixin_registry.register(BAG)
@dataclass
class BagInputFeatureConfigMixin(BaseMarshmallowConfig):
    """BagInputFeatureConfigMixin is a dataclass that configures the parameters used in both the bag input feature
    and the bag global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BAG)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=BAG,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(BAG)
@dataclass(repr=False)
class BagInputFeatureConfig(BaseInputFeatureConfig, BagInputFeatureConfigMixin):
    """BagInputFeatureConfig is a dataclass that configures the parameters used for a bag input feature."""

    pass
