from typing import List

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.schema.features.augmentation.image import RandomHorizontalFlipConfig, RandomRotateConfig
from ludwig.schema.features.augmentation.utils import AugmentationDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import ecd_defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass

# Augmentation operations when augmentation is set to True
AUGMENTATION_DEFAULT_OPERATIONS = [
    RandomHorizontalFlipConfig(),
    RandomRotateConfig(),
]


@DeveloperAPI
@ecd_defaults_config_registry.register(IMAGE)
@input_mixin_registry.register(IMAGE)
@ludwig_dataclass
class ImageInputFeatureConfigMixin(BaseMarshmallowConfig):
    """ImageInputFeatureConfigMixin is a dataclass that configures the parameters used in both the image input
    feature and the image global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=IMAGE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=IMAGE,
        default="stacked_cnn",
    )

    augmentation: List[BaseAugmentationConfig] = AugmentationDataclassField(
        feature_type=IMAGE,
        default=False,
        default_augmentations=AUGMENTATION_DEFAULT_OPERATIONS,
        description="Augmentation operation configuration.",
    )

    def has_augmentation(self) -> bool:
        # Check for None, False, and []
        return bool(self.augmentation)


@DeveloperAPI
@ecd_input_config_registry.register(IMAGE)
@ludwig_dataclass
class ImageInputFeatureConfig(ImageInputFeatureConfigMixin, BaseInputFeatureConfig):
    """ImageInputFeatureConfig is a dataclass that configures the parameters used for an image input feature."""

    type: str = schema_utils.ProtectedString(IMAGE)
