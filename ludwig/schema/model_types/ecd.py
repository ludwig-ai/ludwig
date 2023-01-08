from dataclasses import field
from typing import List, Optional
from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import CombinerSelection
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.model_types.base import BaseModelTypeConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import ECDTrainerConfig


@DeveloperAPI
@register_model_type(name="sgd")
@dataclass(repr=False)
class ECDModelConfig(BaseModelTypeConfig):
    """Parameters for stochastic gradient descent."""

    model_type: str = schema_utils.ProtectedString("ecd")

    input_features: List[BaseInputFeatureConfig] = field(default_factory=list)
    output_features: List[BaseOutputFeatureConfig] = field(default_factory=list)

    combiner: BaseCombinerConfig = CombinerSelection().get_default_field()

    trainer: ECDTrainerConfig = field(default_factory=ECDTrainerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    hyperopt: Optional[HyperoptConfig] = None
