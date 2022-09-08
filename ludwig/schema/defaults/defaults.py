from marshmallow_dataclass import dataclass

from ludwig.constants import AUDIO, BINARY, CATEGORY, DATE, H3, IMAGE, NUMBER, SEQUENCE, SET, TEXT, TIMESERIES, VECTOR
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.utils import DefaultsDataclassField
from ludwig.schema.features.base import BaseFeatureConfig


@dataclass
class DefaultsConfig(schema_utils.BaseMarshmallowConfig):

    audio: BaseFeatureConfig = DefaultsDataclassField(feature_type=AUDIO)

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


def get_defaults_jsonschema():
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    defaults_schema = schema_utils.unload_jsonschema_from_marshmallow_class(DefaultsConfig)
    props = defaults_schema["properties"]
    return {
        "type": "object",
        "properties": props,
    }
