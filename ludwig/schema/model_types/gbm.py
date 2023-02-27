from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.gbm import GBMDefaultsConfig, GBMDefaultsField
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    FeatureCollection,
    GBMInputFeatureSelection,
    GBMOutputFeatureSelection,
)
from ludwig.schema.hyperopt import HyperoptConfig, HyperoptField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import GBMTrainerConfig, GBMTrainerField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_model_type(name="gbm")
@ludwig_dataclass
class GBMModelConfig(ModelConfig):
    """Parameters for GBM."""

    model_type: str = schema_utils.ProtectedString("gbm")

    input_features: FeatureCollection[BaseInputFeatureConfig] = GBMInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = GBMOutputFeatureSelection().get_list_field()

    trainer: GBMTrainerConfig = GBMTrainerField().get_default_field()
    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: GBMDefaultsConfig = GBMDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()
