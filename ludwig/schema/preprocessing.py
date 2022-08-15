from typing import Tuple

from marshmallow_dataclass import dataclass
# from dataclasses import field
#
# from ludwig.constants import (
#     AUDIO,
#     BINARY,
#     CATEGORY,
#     DATE,
#     H3,
#     IMAGE,
#     NUMBER,
#     SEQUENCE,
#     SET,
#     TEXT,
#     TIMESERIES,
#     VECTOR,
# )
from ludwig.schema import utils as schema_utils
# from ludwig.schema.features.base import BasePreprocessingConfig
# from ludwig.schema.features.utils import PreprocessingDataclassField


@dataclass
class PreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Global preprocessing config is a dataclass that configures the parameters used for global preprocessing"""

    force_split: bool = schema_utils.Boolean(
        default=False,
        description="If true, the split column in the dataset file is ignored and the dataset is randomly split. If "
                    "false the split column is used if available. "
    )

    split_probabilities: Tuple[float] = schema_utils.FloatRangeTupleDataclassField(
        n=3,
        default=(0.7, 0.1, 0.2),
        description="The proportion of the dataset data to end up in training, validation and test, respectively. "
                    "The three values must sum to 1.0. "
    )

    stratify: str = schema_utils.String(
        default=None,
        description="If null the split is random, otherwise you can specify the name of a category feature and the "
                    "split will be stratified on that feature. "
    )

    oversample_minority: float = schema_utils.NonNegativeFloat(
        default=None,
        description="If not None, the minority class will be oversampled to reach the specified ratio respective to "
                    "the majority class. "
    )

    undersample_majority: float = schema_utils.NonNegativeFloat(
        default=None,
        description="If not None, the majority class will be undersampled to reach the specified ratio respective "
                    "to the minority class. "
    )

    # audio: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=AUDIO)
    #
    # binary: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=BINARY)
    #
    # category: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=CATEGORY)
    #
    # date: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=DATE)
    #
    # h3: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=H3)
    #
    # image: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=IMAGE)
    #
    # number: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=NUMBER)
    #
    # sequence: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=SEQUENCE)
    #
    # set: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=SET)
    #
    # text: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=TEXT)
    #
    # timeseries: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=TIMESERIES)
    #
    # vector: Optional[BasePreprocessingConfig] = PreprocessingDataclassField(feature_type=VECTOR)


def get_preprocessing_jsonschema():
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    preproc_schema = schema_utils.unload_jsonschema_from_marshmallow_class(PreprocessingConfig)
    props = preproc_schema["properties"]
    return {
        "type": "object",
        "allOf": props,
        "required": ["type"],
    }
