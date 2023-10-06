from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.config_validation.checks import get_config_check_registry
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.external import ExternalDefaultsConfig, ExternalDefaultsField
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    ExternalInputFeatureSelection,
    ExternalOutputFeatureSelection,
    FeatureCollection,
)
from ludwig.schema.hyperopt import HyperoptConfig, HyperoptField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.model_types.utils import set_derived_feature_columns_
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import ExternalTrainerConfig, ExternalTrainerField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_model_type(name="external")
@ludwig_dataclass
class ExternalModelConfig(ModelConfig):
    """Parameters for External Model Type."""

    model_type: str = schema_utils.ProtectedString("external")

    input_features: FeatureCollection[BaseInputFeatureConfig] = ExternalInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = ExternalOutputFeatureSelection().get_list_field()

    preprocessing: Optional[PreprocessingConfig] = PreprocessingField().get_default_field()
    defaults: Optional[ExternalDefaultsConfig] = ExternalDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()

    trainer: Optional[ExternalTrainerConfig] = ExternalTrainerField().get_default_field()

    def __post_init__(self):
        # Derive proc_col for each feature from the feature's preprocessing parameters
        # after all preprocessing parameters have been set
        set_derived_feature_columns_(self)

        # Auxiliary checks.
        get_config_check_registry().check_config(self)
