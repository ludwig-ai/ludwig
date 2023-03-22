from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import CombinerSelection
from ludwig.schema.defaults.ecd import ECDDefaultsConfig, ECDDefaultsField
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    ECDInputFeatureSelection,
    ECDOutputFeatureSelection,
    FeatureCollection,
)
from ludwig.schema.hyperopt import HyperoptConfig, HyperoptField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import ECDTrainerConfig, ECDTrainerField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_model_type(name="ecd")
@ludwig_dataclass
class ECDModelConfig(ModelConfig):
    """Parameters for ECD."""

    model_type: str = schema_utils.ProtectedString("ecd")

    input_features: FeatureCollection[BaseInputFeatureConfig] = ECDInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = ECDOutputFeatureSelection().get_list_field()

    combiner: BaseCombinerConfig = CombinerSelection().get_default_field()

    trainer: ECDTrainerConfig = ECDTrainerField().get_default_field()
    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: ECDDefaultsConfig = ECDDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()
