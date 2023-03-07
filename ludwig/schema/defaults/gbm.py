from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, NUMBER
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.base import BaseDefaultsConfig
from ludwig.schema.defaults.utils import DefaultsDataclassField
from ludwig.schema.features.base import BaseFeatureConfig
from ludwig.schema.features.utils import gbm_defaults_config_registry
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class GBMDefaultsConfig(BaseDefaultsConfig):
    binary: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=BINARY, defaults_registry=gbm_defaults_config_registry
    )

    category: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=CATEGORY, defaults_registry=gbm_defaults_config_registry
    )

    number: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=NUMBER, defaults_registry=gbm_defaults_config_registry
    )


@DeveloperAPI
class GBMDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(GBMDefaultsConfig)
