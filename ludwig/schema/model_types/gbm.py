from dataclasses import field
from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    FeatureCollection,
    GBMInputFeatureSelection,
    GBMOutputFeatureSelection,
)
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import GBMTrainerConfig


@DeveloperAPI
@register_model_type(name="gbm")
@dataclass(repr=False)
class GBMModelConfig(ModelConfig):
    """Parameters for GBM."""

    model_type: str = schema_utils.ProtectedString("gbm")

    input_features: FeatureCollection[BaseInputFeatureConfig] = GBMInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = GBMOutputFeatureSelection().get_list_field()

    trainer: GBMTrainerConfig = field(default_factory=GBMTrainerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    hyperopt: Optional[HyperoptConfig] = None
