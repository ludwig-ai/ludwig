from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, NUMBER
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.base import BaseDefaultsConfig
from ludwig.schema.defaults.utils import DefaultsDataclassField
from ludwig.schema.features.base import BaseFeatureConfig
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class GBMDefaultsConfig(BaseDefaultsConfig):
    binary: BaseFeatureConfig = DefaultsDataclassField(feature_type=BINARY)

    category: BaseFeatureConfig = DefaultsDataclassField(feature_type=CATEGORY)

    number: BaseFeatureConfig = DefaultsDataclassField(feature_type=NUMBER)


@DeveloperAPI
def get_gbm_defaults_jsonschema():
    """Returns a JSON schema structured to only require a `type` key and then conditionally apply a corresponding
    combiner's field constraints."""
    preproc_schema = schema_utils.unload_jsonschema_from_marshmallow_class(GBMDefaultsConfig)
    props = preproc_schema["properties"]

    return {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
        "title": "global_defaults_options",
        "description": "Set global defaults for input and output features",
    }


@DeveloperAPI
class GBMDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(GBMDefaultsConfig)

    @staticmethod
    def _jsonschema_type_mapping():
        return get_gbm_defaults_jsonschema()
