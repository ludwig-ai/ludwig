from marshmallow_dataclass import dataclass
from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils


@DeveloperAPI
@dataclass(repr=False, order=True)
class BaseCombinerConfig(schema_utils.BaseMarshmallowConfig):
    """Base combiner config class."""

    type: str
