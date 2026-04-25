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
from ludwig.schema.llms.base_model import BaseModelDataclassField
from ludwig.schema.llms.generation import LLMGenerationConfig, LLMGenerationConfigField
from ludwig.schema.llms.model_parameters import ModelParametersConfig, ModelParametersConfigField
from ludwig.schema.llms.peft import (
    AdapterDataclassField,
    BaseAdapterConfig,
    NamedAdaptersConfig,
    NamedAdaptersDataclassField,
)
from ludwig.schema.llms.prompt import PromptConfig, PromptConfigField
from ludwig.schema.llms.quantization import QuantizationConfig, QuantizationConfigField
from ludwig.schema.model_types.base import ModelConfig, register_model_type
from ludwig.schema.preprocessing import PreprocessingConfig, PreprocessingField
from ludwig.schema.trainer import LLMTrainerConfig, LLMTrainerDataclassField


@DeveloperAPI
@register_model_type(name="llm")
class LLMModelConfig(ModelConfig):
    """Parameters for LLM Model Type."""

    model_type: str = schema_utils.ProtectedString("llm")

    base_model: str = BaseModelDataclassField()

    input_features: FeatureCollection[BaseInputFeatureConfig] = LLMInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = LLMOutputFeatureSelection().get_list_field()

    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: LLMDefaultsConfig | None = LLMDefaultsField().get_default_field()
    hyperopt: HyperoptConfig | None = HyperoptField().get_default_field()

    prompt: PromptConfig = PromptConfigField().get_default_field()

    # trainer: LLMTrainerConfig = LLMTrainerField().get_default_field()
    trainer: LLMTrainerConfig = LLMTrainerDataclassField(
        description="The trainer to use for the model",
    )

    generation: LLMGenerationConfig = LLMGenerationConfigField().get_default_field()

    adapter: BaseAdapterConfig | None = AdapterDataclassField()
    adapters: NamedAdaptersConfig | None = NamedAdaptersDataclassField()
    quantization: QuantizationConfig | None = QuantizationConfigField().get_default_field()
    model_parameters: ModelParametersConfig | None = ModelParametersConfigField().get_default_field()

    trust_remote_code: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Whether to trust and execute remote code from the HuggingFace model repository. "
            "Required for some models (e.g. Phi-2, Qwen) that use custom architectures. "
            "Only enable this for models you trust."
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        # `adapter:` (singular) and `adapters:` (plural) are mutually exclusive.
        # A config must use one form or the other — using both is ambiguous because the
        # model would have both an anonymous adapter and a registry of named ones.
        if self.adapter is not None and self.adapters is not None:
            raise ConfigValidationError(
                "Cannot set both `adapter:` and `adapters:` — use one form or the other. "
                "Use `adapter:` for a single adapter (common case). Use `adapters:` to "
                "register multiple named adapters that can be switched at runtime or merged."
            )
        if self.adapters is not None:
            if not self.adapters.adapters:
                raise ConfigValidationError(
                    "`adapters.adapters` must contain at least one entry. To disable "
                    "parameter-efficient fine-tuning, remove the `adapters:` field entirely."
                )
            adapter_names = list(self.adapters.adapters.keys())
            if self.adapters.active is not None and self.adapters.active not in adapter_names:
                # The active adapter name may also refer to a merged adapter defined below.
                if not (self.adapters.merge and self.adapters.merge.name == self.adapters.active):
                    raise ConfigValidationError(
                        f"`adapters.active` = {self.adapters.active!r} does not match any "
                        f"adapter name in `adapters.adapters` ({adapter_names}) or the "
                        "`adapters.merge.name` field."
                    )
            if self.adapters.merge is not None:
                merge = self.adapters.merge
                if not merge.sources:
                    raise ConfigValidationError("`adapters.merge.sources` is required when `adapters.merge` is set.")
                unknown = [s for s in merge.sources if s not in adapter_names]
                if unknown:
                    raise ConfigValidationError(
                        f"`adapters.merge.sources` references unknown adapter names: {unknown}. "
                        f"Known adapters: {adapter_names}."
                    )
                if merge.weights is not None and len(merge.weights) != len(merge.sources):
                    raise ConfigValidationError(
                        f"`adapters.merge.weights` has {len(merge.weights)} entries but "
                        f"`adapters.merge.sources` has {len(merge.sources)}. Lengths must match."
                    )
                if merge.name in adapter_names:
                    raise ConfigValidationError(
                        f"`adapters.merge.name` = {merge.name!r} collides with an existing "
                        "source adapter name. Pick a different name for the merged adapter."
                    )
