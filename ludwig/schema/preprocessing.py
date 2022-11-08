from marshmallow_dataclass import dataclass

from ludwig.constants import RANDOM
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.schema.split import BaseSplitConfig, SplitDataclassField


@dataclass(repr=False, order=True)
class PreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Global preprocessing config is a dataclass that configures the parameters used for global preprocessing."""

    sample_ratio: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="The ratio of the dataset to use. For instance, if 0.5, half of the dataset "
        "provided will be used.",
        parameter_metadata=PREPROCESSING_METADATA["sample_ratio"],
    )

    oversample_minority: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="If not None, the minority class will be oversampled to reach the specified ratio respective to "
        "the majority class. ",
        parameter_metadata=PREPROCESSING_METADATA["oversample_minority"],
    )

    undersample_majority: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="If not None, the majority class will be undersampled to reach the specified ratio respective "
        "to the minority class. ",
        parameter_metadata=PREPROCESSING_METADATA["undersample_majority"],
    )

    split: BaseSplitConfig = SplitDataclassField(
        default=RANDOM,
    )


def get_preprocessing_jsonschema():
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    preproc_schema = schema_utils.unload_jsonschema_from_marshmallow_class(PreprocessingConfig)
    props = preproc_schema["properties"]
    return {
        "type": "object",
        "properties": props,
        "additionalProperties": True,
        "title": "global_preprocessing_options",
        "description": "Select the preprocessing type.",
    }
