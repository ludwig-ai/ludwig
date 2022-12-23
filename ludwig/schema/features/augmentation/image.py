from marshmallow_dataclass import dataclass
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUGMENTATION, IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.schema.features.augmentation.utils import register_augmentation_config
from ludwig.schema.metadata.feature_metadata import FEATURE_METADATA


@DeveloperAPI
@register_augmentation_config(name="random_horizontal_flip")
@dataclass(repr=False)
class RandomHorizontalFlipConfig(BaseAugmentationConfig):
    """Random horizontal flip augmentation operation."""

    pass


@DeveloperAPI
@register_augmentation_config(name="random_vertical_flip")
@dataclass(repr=False)
class RandomVerticalFlipConfig(BaseAugmentationConfig):
    """Random vertical flip augmentation operation."""

    pass


@DeveloperAPI
@register_augmentation_config(name="random_rotate")
@dataclass(repr=False)
class RandomRotateConfig(BaseAugmentationConfig):
    """Random rotation augmentation operation."""

    degree: int = schema_utils.Integer(
        default=45,
        description="Range of angle for random rotation, i.e.,  [-degree, +degree].",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["rotation_degree"],
    )

# class RandomBlurOperation(BaseAugmentationOperationConfig):
#     """Random blur augmentation operation."""
#
#     max_kernel_size: int = schema_utils.Integer(
#         default=3,
#         description="Maximum kernel size for random blur.",
#         parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["max_kernel_size"],
#     )
#
#
# @register_augmentation
# class RandomBrightnessOperation(BaseAugmentationOperationConfig):
#     """Random brightness augmentation operation."""
#
#     max_delta: int = schema_utils.Integer(
#         default=0.5,
#         description="Maximum delta for random brightness.",
#         parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["max_delta"],
#     )
#
#     min_delta: int = schema_utils.Integer(
#         default=0.5,
#         description="Minimum delta for random brightness.",
#         parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["min_delta"],
#     )
