from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.llm import LLMDefaultsConfig, LLMDefaultsField
from ludwig.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    FeatureCollection,
    LLMInputFeatureSelection,
    LLMOutputFeatureSelection,
)
from ludwig.schema.hyperopt import HyperoptConfig, HyperoptField
from ludwig.schema.llms.base_model import BaseModelConfig, BaseModelDataclassField
from ludwig.schema.llms.generation import LLMGenerationConfig, LLMGenerationConfigField
from ludwig.schema.llms.peft import AdapterDataclassField, BaseAdapterConfig
from ludwig.schema.llms.prompt import PromptConfig, PromptConfigField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import LLMTrainerConfig, LLMTrainerDataclassField
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_model_type(name="llm")
@ludwig_dataclass
class LLMModelConfig(ModelConfig):
    """Parameters for LLM Model Type."""

    def __post_init__(self):
        if self.base_model is None:
            raise ConfigValidationError(
                "LLM requires `base_model` to be set. This can be a preset or any pretrained CausalLM on huggingface. "
                "See: https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads"
            )

        super().__post_init__()

    model_type: str = schema_utils.ProtectedString("llm")

    base_model: BaseModelConfig = BaseModelDataclassField(
        default=None,
        description=(
            "Base pretrained model to use. This can be one of the presets defined by Ludwig, a fully qualified "
            "name of a pretrained model from the HuggingFace Hub, or a path to a directory containing a "
            "pretrained model."
        ),
    )

    input_features: FeatureCollection[BaseInputFeatureConfig] = LLMInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = LLMOutputFeatureSelection().get_list_field()

    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: Optional[LLMDefaultsConfig] = LLMDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()

    prompt: PromptConfig = PromptConfigField().get_default_field()

    # trainer: LLMTrainerConfig = LLMTrainerField().get_default_field()
    trainer: LLMTrainerConfig = LLMTrainerDataclassField(
        description="The trainer to use for the model",
    )

    generation: LLMGenerationConfig = LLMGenerationConfigField().get_default_field()

    adapter: Optional[BaseAdapterConfig] = AdapterDataclassField(
        description="The parameter-efficient finetuning strategy to use for the model"
    )
