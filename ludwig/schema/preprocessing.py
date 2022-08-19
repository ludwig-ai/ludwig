from typing import Tuple, Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import (
    AUDIO,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    NUMBER,
    RANDOM,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.split import BaseSplitConfig, SplitDataclassField
from ludwig.schema.features.base import BasePreprocessingConfig
from ludwig.schema.features.utils import PreprocessingDataclassField


@dataclass
class PreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """Global preprocessing config is a dataclass that configures the parameters used for global preprocessing"""

    split: BaseSplitConfig = SplitDataclassField(
        default=RANDOM,
    )

    sample_ratio: float = schema_utils.NonNegativeFloat(
        default=1.0,
        description="Ratio of the dataset to use for training. If 1.0, all the data is used for training."
    )

    oversample_minority: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="If not None, the minority class will be oversampled to reach the specified ratio respective to "
                    "the majority class. "
    )

    undersample_majority: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
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
