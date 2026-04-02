from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils


@DeveloperAPI
class BaseCombinerConfig(schema_utils.LudwigBaseConfig):
    """Base combiner config class."""

    type: str
