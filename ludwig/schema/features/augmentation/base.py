from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseAugmentationConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for augmentation."""

    type: str
