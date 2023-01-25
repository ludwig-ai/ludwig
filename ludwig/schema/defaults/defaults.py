from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    MODEL_ECD,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.utils import DefaultsDataclassField
from ludwig.schema.features.base import BaseFeatureConfig


@DeveloperAPI
@dataclass
class DefaultsConfig(schema_utils.BaseMarshmallowConfig):
    audio: BaseFeatureConfig = DefaultsDataclassField(feature_type=AUDIO)

    bag: BaseFeatureConfig = DefaultsDataclassField(feature_type=BAG)

    binary: BaseFeatureConfig = DefaultsDataclassField(feature_type=BINARY)

    category: BaseFeatureConfig = DefaultsDataclassField(feature_type=CATEGORY)

    date: BaseFeatureConfig = DefaultsDataclassField(feature_type=DATE)

    h3: BaseFeatureConfig = DefaultsDataclassField(feature_type=H3)

    image: BaseFeatureConfig = DefaultsDataclassField(feature_type=IMAGE)

    number: BaseFeatureConfig = DefaultsDataclassField(feature_type=NUMBER)

    sequence: BaseFeatureConfig = DefaultsDataclassField(feature_type=SEQUENCE)

    set: BaseFeatureConfig = DefaultsDataclassField(feature_type=SET)

    text: BaseFeatureConfig = DefaultsDataclassField(feature_type=TEXT)

    timeseries: BaseFeatureConfig = DefaultsDataclassField(feature_type=TIMESERIES)

    vector: BaseFeatureConfig = DefaultsDataclassField(feature_type=VECTOR)


@dataclass
class GBMDefaultsConfig(schema_utils.BaseMarshmallowConfig):
    binary: BaseFeatureConfig = DefaultsDataclassField(feature_type=BINARY)

    category: BaseFeatureConfig = DefaultsDataclassField(feature_type=CATEGORY)

    number: BaseFeatureConfig = DefaultsDataclassField(feature_type=NUMBER)


@DeveloperAPI
def get_defaults_jsonschema(model_type: str = MODEL_ECD):
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    cls = DefaultsConfig if model_type == MODEL_ECD else GBMDefaultsConfig
    preproc_schema = schema_utils.unload_jsonschema_from_marshmallow_class(cls)
    props = preproc_schema["properties"]
    return {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
        "title": "global_defaults_options",
        "description": "Set global defaults for input and output features",
    }
