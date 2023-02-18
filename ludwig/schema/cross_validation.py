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


@DeveloperAPI
def get_cv_jsonschema():
    props = schema_utils.unload_jsonschema_from_marshmallow_class(CrossValidationField)["properties"]
    return {
        "type": ["object", "null"],
        "properties": props,
        "title": "cross_validation_options",
        "description": "Settings for cross validation",
    }


@DeveloperAPI
class CrossValidationField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(CrossValidationField, default_missing=True)

    @staticmethod
    def _jsonschema_type_mapping():
        return get_cv_jsonschema()
