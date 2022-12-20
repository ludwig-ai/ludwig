
from marshmallow_dataclass import dataclass
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUGMENTATION, IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.augmentation.base import BaseAugmentationConfig
from ludwig.schema.features.augmentation.utils import register_augmentation  # TODO: Is this needed?
from ludwig.schema.metadata.feature_metadata import FEATURE_METADATA


# TODO: Is this needed?
@DeveloperAPI
@register_augmentation(IMAGE)
@dataclass
class ImageAugmentationConfig(BaseAugmentationConfig):

    random_vertical_flip: bool = schema_utils.Boolean(
        default=False,
        description="If true, then image will be randomly flipped vertically.",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["random_vertical_flip"],
    )
