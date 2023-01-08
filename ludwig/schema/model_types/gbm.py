from dataclasses import field
from typing import List, Optional
from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.model_types.base import BaseModelTypeConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import GBMTrainerConfig


@DeveloperAPI
@register_model_type(name="gbm")
@dataclass(repr=False)
class GBMModelConfig(BaseModelTypeConfig):
    """Parameters for GBM."""

    model_type: str = schema_utils.ProtectedString("gbm")

    input_features: List[BaseInputFeatureConfig] = field(default_factory=list)
    output_features: List[BaseOutputFeatureConfig] = field(default_factory=list)

    trainer: GBMTrainerConfig = GBMTrainerConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    hyperopt: Optional[HyperoptConfig] = None
