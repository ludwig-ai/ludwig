from abc import ABC

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class CrossValidationConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Basic cross validation settings."""

    num_folds: int = schema_utils.PositiveInteger(
        default=10,
        description="Number of folds (k) used during k-fold cross validation.",
    )
