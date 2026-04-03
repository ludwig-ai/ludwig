from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils


@DeveloperAPI
class BaseDefaultsConfig(schema_utils.LudwigBaseConfig):
    """Base defaults config class."""
