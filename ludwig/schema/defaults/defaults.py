from marshmallow_dataclass import dataclass

from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    DECODER,
    ENCODER,
    H3,
    IMAGE,
    LOSS,
    NUMBER,
    PREPROCESSING,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.utils import DefaultsDataclassField
from ludwig.schema.features.base import BaseFeatureConfig


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

    def to_dict(self):
        """This method overwrites the default to_dict method for getting a dictionary representation of this dataclass
            because we need to remove excess parameters that cannot be removed from the dataclass itself.

        Returns: dict for this dataclass
        """
        output_dict = schema_utils.convert_submodules(self.__dict__)

        for feature_type in output_dict.keys():
            output_dict[feature_type] = {key: val for key, val in output_dict[feature_type].items()
                                         if key in [ENCODER, PREPROCESSING, DECODER, LOSS]}

        return output_dict


def get_defaults_jsonschema():
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    return schema_utils.unload_jsonschema_from_marshmallow_class(DefaultsConfig, additional_properties=False)
