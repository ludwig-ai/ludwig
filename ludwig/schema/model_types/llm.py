from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.ecd import ECDDefaultsConfig, ECDDefaultsField
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    FeatureCollection,
    LLMInputFeatureSelection,
    LLMOutputFeatureSelection,
)
from ludwig.schema.hyperopt import HyperoptConfig, HyperoptField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import LLMTrainerConfig, LLMTrainerField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_model_type(name="llm")
@ludwig_dataclass
class LLMModelConfig(ModelConfig):
    """Parameters for LLM Model Type."""

    model_type: str = schema_utils.ProtectedString("llm")

    model_name: str = schema_utils.String(
        default="",
        allow_none=False,
        description=(
            "The name of the model to use. This can be a local path or a "
            "remote path. If it is a remote path, it must be a valid HuggingFace "
            "model name. If it is a local path, it must be a valid HuggingFace "
            "model name or a path to a local directory containing a valid "
            "HuggingFace model."
        ),
    )

    generation_config: dict = schema_utils.Dict(
        default={},
        allow_none=False,
        description=(
            "The generation config to use for the model. This is a dictionary "
            "that will be passed to the `generate` method of the HuggingFace "
            "model. See the HuggingFace documentation for more details."
        ),
    )

    input_features: FeatureCollection[BaseInputFeatureConfig] = LLMInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = LLMOutputFeatureSelection().get_list_field()

    trainer: LLMTrainerConfig = LLMTrainerField().get_default_field()
    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: ECDDefaultsConfig = ECDDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()
