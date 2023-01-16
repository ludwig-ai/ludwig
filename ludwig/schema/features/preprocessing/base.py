from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import PREPROCESSING_METADATA


@DeveloperAPI
class BasePreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for input feature preprocessing. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding input feature class are copied over: check each class to check which attributes are different
    from the preprocessing of each feature.
    """

    cache_encoder_embeddings: bool = schema_utils.Boolean(
        default=False,
        description="Compute encoder embeddings in preprocessing, speeding up training time considerably.",
        parameter_metadata=PREPROCESSING_METADATA["cache_encoder_embeddings"],
    )
