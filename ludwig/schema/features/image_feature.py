from typing import List

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE, LOSS, MODEL_ECD, SOFTMAX_CROSS_ENTROPY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.schema.features.augmentation.image import RandomHorizontalFlipConfig, RandomRotateConfig
from ludwig.schema.features.augmentation.utils import AugmentationDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.utils import LossDataclassField
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import (
    ecd_defaults_config_registry,
    ecd_input_config_registry,
    ecd_output_config_registry,
    input_mixin_registry,
    output_mixin_registry,
)
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
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


@DeveloperAPI
@output_mixin_registry.register(IMAGE)
@ludwig_dataclass
class ImageOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """ImageOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the image output
    feature and the image global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=IMAGE,
        default="unet",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=IMAGE,
        default=SOFTMAX_CROSS_ENTROPY,
    )


@DeveloperAPI
@ecd_output_config_registry.register(IMAGE)
@ludwig_dataclass
class ImageOutputFeatureConfig(ImageOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """ImageOutputFeatureConfig is a dataclass that configures the parameters used for an image output feature."""

    type: str = schema_utils.ProtectedString(IMAGE)

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[IMAGE]["dependencies"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [LOSS],
        default=LOSS,
        description="Internal only use parameter: default validation metric for image output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="image_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[IMAGE]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[IMAGE]["reduce_input"],
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(IMAGE)
@ludwig_dataclass
class ImageDefaultsConfig(ImageInputFeatureConfigMixin, ImageOutputFeatureConfigMixin):
    pass
