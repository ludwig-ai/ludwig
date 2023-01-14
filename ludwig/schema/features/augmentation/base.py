from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils


@DeveloperAPI
class BaseAugmentationConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for augmentation."""

    pass
