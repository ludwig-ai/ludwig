from typing import Dict

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
    MODEL_GBM,
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


def prune_gbm_props(schema: Dict):
    """Removes unsupported props from the given JSON schema.

    Designed for use with `get_defaults_jsonschema`.
    """
    gbm_feature_types = ["binary", "category", "number"]
    pruned_props = {}
    for default_feature_type in schema["properties"]:
        if default_feature_type in gbm_feature_types:
            ft_schema = schema["properties"][default_feature_type]
            pruned_ft_schema_properties = {
                k: v for k, v in ft_schema["properties"].items() if k not in ["encoder", "decoder", "tied"]
            }
            pruned_props[default_feature_type] = {**ft_schema, "properties": pruned_ft_schema_properties}
    return {**schema, "properties": pruned_props}


@DeveloperAPI
def get_defaults_jsonschema(model_type: str = MODEL_ECD):
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    preproc_schema = schema_utils.unload_jsonschema_from_marshmallow_class(DefaultsConfig)

    # Prune extra props if is GBM model:
    if model_type == MODEL_GBM:
        preproc_schema = prune_gbm_props(preproc_schema)

    return {
        "type": "object",
        "properties": preproc_schema["properties"],
        "additionalProperties": False,
        "title": "global_defaults_options",
        "description": "Set global defaults for input and output features",
    }
