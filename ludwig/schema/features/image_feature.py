from typing import List

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.schema.features.augmentation.utils import AugmentationContainerDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import input_config_registry, input_mixin_registry


@DeveloperAPI
@input_mixin_registry.register(IMAGE)
@dataclass
class ImageInputFeatureConfigMixin(schema_utils.BaseMarshmallowConfig):
    """ImageInputFeatureConfigMixin is a dataclass that configures the parameters used in both the image input
    feature and the image global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=IMAGE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=IMAGE,
        default="stacked_cnn",
    )

    augmentation: List[BaseAugmentationConfig] = AugmentationContainerDataclassField(
        feature_type=IMAGE,
        default=[
            {"type": "random_horizontal_flip"},
        ]
    )


@DeveloperAPI
@input_config_registry.register(IMAGE)
@dataclass(repr=False)
class ImageInputFeatureConfig(BaseInputFeatureConfig, ImageInputFeatureConfigMixin):
    """ImageInputFeatureConfig is a dataclass that configures the parameters used for an image input feature."""

    pass
