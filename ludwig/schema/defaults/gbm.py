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
class GBMDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(GBMDefaultsConfig)
