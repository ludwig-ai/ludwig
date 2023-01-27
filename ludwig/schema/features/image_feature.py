from typing import List, Union

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
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@input_mixin_registry.register(IMAGE)
@ludwig_dataclass
class ImageInputFeatureConfigMixin(BaseMarshmallowConfig):
    """ImageInputFeatureConfigMixin is a dataclass that configures the parameters used in both the image input
    feature and the image global defaults section of the Ludwig Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=IMAGE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=IMAGE,
        default="stacked_cnn",
    )

    augmentation: Union[bool, List[BaseAugmentationConfig]] = schema_utils.OneOfOptionsField(
        default=False,
        description="Augmentation configuration.",
        field_options=[
            schema_utils.Boolean(
                default=False,
                description="Whether to use augmentation or not.",
            ),
            AugmentationContainerDataclassField(
                feature_type=IMAGE,
                description="Augmentation operation configuration.",
            ),
        ],
    )


@DeveloperAPI
@input_config_registry.register(IMAGE)
@ludwig_dataclass
class ImageInputFeatureConfig(BaseInputFeatureConfig, ImageInputFeatureConfigMixin):
    """ImageInputFeatureConfig is a dataclass that configures the parameters used for an image input feature."""

    pass
