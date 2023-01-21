from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils


@DeveloperAPI
@dataclass(repr=False)
class BaseAugmentationConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for augmentation."""

    pass
