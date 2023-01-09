from dataclasses import field
from typing import Optional
from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import CombinerSelection
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    ECDInputFeatureSelection,
    ECDOutputFeatureSelection,
    FeatureCollection,
)
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.model_types.base import BaseModelTypeConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import ECDTrainerConfig


@DeveloperAPI
@register_model_type(name="ecd")
@dataclass(repr=False)
class ECDModelConfig(BaseModelTypeConfig):
    """Parameters for ECD."""

    model_type: str = schema_utils.ProtectedString("ecd")

    input_features: FeatureCollection[BaseInputFeatureConfig] = ECDInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = ECDOutputFeatureSelection().get_list_field()

    combiner: BaseCombinerConfig = CombinerSelection().get_default_field()

    trainer: ECDTrainerConfig = field(default_factory=ECDTrainerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    hyperopt: Optional[HyperoptConfig] = None
