from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.utils.registry import Registry

if TYPE_CHECKING:
    from peft import PeftConfig


adapter_registry = Registry()


@DeveloperAPI
def register_adapter(name: str):
    def wrap(config: BaseAdapterConfig):
        adapter_registry[name] = config
        return config

    return wrap


@DeveloperAPI
class LoraPostprocessorConfig(schema_utils.LudwigBaseConfig):
    """This Dataclass is a schema for the nested postprocessing config under adapter of type "lora"."""

    merge_adapter_into_base_model: bool = schema_utils.Boolean(
        default=False,
        description="""Instructs whether or not the fine-tuned LoRA weights are to be merged into the base LLM model so
that the complete fine-tuned model is available to be used and/or persisted, and then reused upon loading as a single
model (rather than having to load base and fine-tuned models separately).""",
    )
    progressbar: bool = schema_utils.Boolean(
        default=False,
        description="Instructs whether or not to show a progress bar indicating the unload and merge process.",
    )


@DeveloperAPI
class LoraPostprocessorConfigField(schema_utils.NestedConfigField):
    def __init__(self):
        super().__init__(LoraPostprocessorConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_config_class(LoraPostprocessorConfig, title="LoraPostprocessor")


@DeveloperAPI
class BaseAdapterConfig(schema_utils.LudwigBaseConfig, ABC):
    type: str

    pretrained_adapter_weights: str | None = schema_utils.String(
        default=None, description="Path to pretrained weights.", allow_none=True
    )

    postprocessor: LoraPostprocessorConfig = LoraPostprocessorConfigField().get_default_field()

    @abstractmethod
    def to_config(self, **kwargs) -> "PeftConfig":
        pass


@DeveloperAPI
@register_adapter(name="lora")
class LoraConfig(BaseAdapterConfig):
    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.r * 2

    type: str = schema_utils.ProtectedString(
        "lora",
        description=LLM_METADATA["adapter"]["lora"]["type"].long_description,
    )

    r: int = schema_utils.PositiveInteger(
        default=8,
        description="Lora attention dimension.",
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["r"],
    )

    alpha: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The alpha parameter for Lora scaling. Defaults to `2 * r`.",
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["alpha"],
    )

    dropout: float = schema_utils.NonNegativeFloat(
        default=0.05,
        description="The dropout probability for Lora layers.",
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["dropout"],
    )

    # TODO(travis): figure out why calling this `bias` doesn't work
    bias_type: str = schema_utils.StringOptions(
        options=["none", "all", "lora_only"],
        default="none",
        description="Bias type for Lora.",
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description=(
            "List of module names or regex expression of the module names to replace with LoRA. "
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
            "Defaults to targeting the query and value matrices of all self-attention and encoder-decoder attention "
            "layers."
        ),
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["target_modules"],
    )

    use_rslora: bool = schema_utils.Boolean(
        default=False,
        description=(
            "When set to True, uses Rank-Stabilized LoRA which sets the adapter scaling factor to "
            "lora_alpha/math.sqrt(r), since it was proven to work better. Otherwise, it will use the original "
            "default value of lora_alpha/r. Paper: https://arxiv.org/abs/2312.03732."
        ),
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["use_rslora"],
    )

    use_dora: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the "
            "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports non-quantized linear layers. DoRA introduces a "
            "bigger overhead than pure LoRA, so it is recommended to merge weights for inference. For more "
            "information, see https://arxiv.org/abs/2402.09353"
        ),
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["use_dora"],
    )

    loraplus_lr_ratio: float | None = schema_utils.Float(
        default=None,
        allow_none=True,
        description=(
            "LoRA+ learning rate ratio (Hayou et al., ICML 2024). When set, the B matrices use "
            "lr * loraplus_lr_ratio while A matrices use the base lr. Typical values: 2-16. "
            "Provides 1-2%% accuracy gain and up to 2x speedup over standard LoRA. "
            "Paper: https://arxiv.org/abs/2402.12354"
        ),
    )

    def to_config(self, task_type: str = None, **kwargs) -> "PeftConfig":
        from peft import LoraConfig as _LoraConfig

        return _LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias_type,
            target_modules=self.target_modules,
            task_type=task_type,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )

    @classmethod
    def name(cls) -> str:
        return "LoRA"

    @classmethod
    def description(cls) -> str:
        return LLM_METADATA["adapter"]["lora"]["type"].long_description


@DeveloperAPI
class BasePromptLearningConfig(BaseAdapterConfig):
    """Config for prompt learning adapters. Not meant to be used directly.

    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/utils/config.py (PromptLearningConfig)
    """

    num_virtual_tokens: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of virtual tokens to add to the prompt. Virtual tokens are used to control the behavior of "
        " the model during inference. ",
        parameter_metadata=LLM_METADATA["adapter"]["prompt_learning"]["num_virtual_tokens"],
    )

    token_dim: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The hidden embedding dimension of the base transformer model.",
    )

    num_transformer_submodules: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of transformer submodules in the base transformer model.",
    )

    num_attention_heads: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of attention heads in the base transformer model.",
    )

    num_layers: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of layers in the base transformer model.",
    )


# TODO(travis): fix text generation when using prompt tuning:
#     RuntimeError: shape '[-1, 17]' is invalid for input of size 9
# @DeveloperAPI
# @register_adapter("prompt_tuning")
# class PromptTuningConfig(BasePromptLearningConfig):
#     """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/prompt_tuning.py."""

#     def __post_init__(self):
#         if self.prompt_tuning_init == "TEXT" and not self.prompt_tuning_init_text:
#             raise ConfigValidationError(
#                 "Must provide `prompt_tuning_init_text` when `prompt_tuning_init` is set to `TEXT`."
#             )

"""#     type: str = schema_utils.ProtectedString("prompt_tuning")"""  # Quotes allow mypy to run without syntax errors.

#     prompt_tuning_init: str = schema_utils.StringOptions(
#         ["RANDOM", "TEXT"],
#         default="RANDOM",
#         description="The type of initialization to use for the prompt embedding. ",
#         parameter_metadata=LLM_METADATA["adapter"]["prompt_tuning"]["prompt_tuning_init"],
#     )

#     prompt_tuning_init_text: str = schema_utils.String(
#         default="",
#         description="The text to use to initialize the prompt embedding.",
#         parameter_metadata=LLM_METADATA["adapter"]["prompt_tuning"]["prompt_tuning_init_text"],
#     )

#     def to_config(self, **kwargs) -> "PeftConfig":
#         from peft import PromptTuningConfig as _PromptTuningConfig

#         return _PromptTuningConfig(
#             num_virtual_tokens=self.num_virtual_tokens,
#             token_dim=self.token_dim,
#             num_transformer_submodules=self.num_transformer_submodules,
#             num_attention_heads=self.num_attention_heads,
#             num_layers=self.num_layers,
#             prompt_tuning_init=self.prompt_tuning_init,
#             prompt_tuning_init_text=self.prompt_tuning_init_text,
#             **kwargs
#         )


# TODO(travis): fix prefix tuning and p-tuning to work with distributed training
# @DeveloperAPI
# @register_adapter("prefix_tuning")
# class PrefixTuningConfig(BasePromptLearningConfig):
#     """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/prefix_tuning.py."""

"""#     type: str = schema_utils.ProtectedString("prefix_tuning")"""  # Quotes allow mypy to run without syntax errors.

#     encoder_hidden_size: Optional[int] = schema_utils.Integer(
#         default=None,
#         allow_none=True,
#         description="The hidden embedding dimension of the prompt encoder.",
#     )

#     prefix_projection: bool = schema_utils.Boolean(
#         default=False,
#         description="Whether to use a projection layer in the prompt encoder to project the prefix tokens",
#     )

#     def to_config(self, task_type: str = None, **kwargs) -> "PeftConfig":
#         from peft import PrefixTuningConfig as _PrefixTuningConfig

#         return _PrefixTuningConfig(
#             num_virtual_tokens=self.num_virtual_tokens,
#             token_dim=self.token_dim,
#             num_transformer_submodules=self.num_transformer_submodules,
#             num_attention_heads=self.num_attention_heads,
#             num_layers=self.num_layers,
#             encoder_hidden_size=self.encoder_hidden_size,
#             prefix_projection=self.prefix_projection,
#             task_type=task_type,
#         )


# @DeveloperAPI
# @register_adapter("p_tuning")
# class PTuningConfig(BasePromptLearningConfig):
"""#     type: str = schema_utils.ProtectedString("p_tuning")"""  # Quotes allow mypy to run without syntax errors.

#     encoder_reparameterization_type: str = schema_utils.StringOptions(
#         ["MLP", "LSTM"],
#         default="MLP",
#         allow_none=False,
#         description="The type of reparameterization to use for the prompt encoder.",
#     )

#     encoder_hidden_size: Optional[int] = schema_utils.PositiveInteger(
#         default=None,
#         allow_none=True,
#         description="The hidden embedding dimension of the prompt encoder.",
#     )

#     encoder_num_layers: Optional[int] = schema_utils.PositiveInteger(
#         default=2,
#         allow_none=True,
#         description="The number of layers in the prompt encoder.",
#     )

#     encoder_dropout: Optional[float] = schema_utils.FloatRange(
#         default=0.0,
#         min=0.0,
#         max=1.0,
#         description="The dropout probability for the prompt encoder.",
#     )

#     def to_config(self, task_type: str = None, **kwargs) -> "PeftConfig":
#         from peft import PromptEncoderConfig as _PromptEncoderConfig

#         return _PromptEncoderConfig(
#             num_virtual_tokens=self.num_virtual_tokens,
#             token_dim=self.token_dim,
#             num_transformer_submodules=self.num_transformer_submodules,
#             num_attention_heads=self.num_attention_heads,
#             num_layers=self.num_layers,
#             encoder_reparameterization_type=self.encoder_reparameterization_type,
#             encoder_hidden_size=self.encoder_hidden_size,
#             encoder_num_layers=self.encoder_num_layers,
#             encoder_dropout=self.encoder_dropout,
#             task_type=task_type,
#         )


@DeveloperAPI
@register_adapter("adalora")
class AdaloraConfig(LoraConfig):
    """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora.py."""

    type: str = schema_utils.ProtectedString(
        "adalora",
        description=LLM_METADATA["adapter"]["adalora"]["type"].long_description,
    )

    target_r: int = schema_utils.PositiveInteger(
        default=8,
        description="Target Lora Matrix Dimension. The target average rank of incremental matrix.",
    )

    init_r: int = schema_utils.PositiveInteger(
        default=12,
        description="Initial Lora Matrix Dimension. The initial rank for each incremental matrix.",
    )

    tinit: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The steps of initial fine-tuning warmup.",
    )

    tfinal: int = schema_utils.NonNegativeInteger(
        default=0,
        description="The steps of final fine-tuning warmup.",
    )

    delta_t: int = schema_utils.NonNegativeInteger(
        default=1,
        description="The time internval between two budget allocations. The step interval of rank allocation.",
    )

    beta1: float = schema_utils.FloatRange(
        default=0.85,
        min=0.0,
        max=1.0,
        description="The hyperparameter of EMA for sensitivity smoothing.",
    )

    beta2: float = schema_utils.FloatRange(
        default=0.85,
        min=0.0,
        max=1.0,
        description=" The hyperparameter of EMA for undertainty quantification.",
    )

    orth_reg_weight: float = schema_utils.FloatRange(
        default=0.5,
        min=0.0,
        max=1.0,
        description="The coefficient of orthogonality regularization.",
    )

    total_step: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=False,
        description="The total training steps for AdaLoRA rank allocation scheduling. "
        "Must be a positive integer (required by peft >= 0.14).",
    )

    rank_pattern: dict | None = schema_utils.Dict(
        default=None,
        allow_none=True,
        description="The allocated rank for each weight matrix by RankAllocator.",
    )

    def to_config(self, **kwargs) -> "PeftConfig":
        from peft import AdaLoraConfig as _AdaLoraConfig

        return _AdaLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias_type,
            target_r=self.target_r,
            init_r=self.init_r,
            tinit=self.tinit,
            tfinal=self.tfinal,
            deltaT=self.delta_t,
            beta1=self.beta1,
            beta2=self.beta2,
            orth_reg_weight=self.orth_reg_weight,
            total_step=self.total_step,
            rank_pattern=self.rank_pattern,
        )

    @classmethod
    def name(cls) -> str:
        return "AdaLoRA"

    @classmethod
    def description(cls) -> str:
        return LLM_METADATA["adapter"]["adalora"]["type"].long_description


@DeveloperAPI
# TODO: <Alex>02/21/2024: Disabling AdaptionPrompt (waiting for PEFT release to fix
# "TypeError: LlamaRotaryEmbedding.forward() missing 1 required positional argument: 'position_ids')"
# (this is reflected in https://github.com/ludwig-ai/ludwig/issues/3938).
# </Alex>
# @register_adapter("adaption_prompt")
class AdaptionPromptConfig(BaseAdapterConfig):
    """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/adaption_prompt/config.py."""

    def __post_init__(self):
        if not self.adapter_len:
            raise ConfigValidationError(
                "`adapter_len` must be set to a value greater than 0 when finetuning is enabled and the adapter"
                "type is `adaption_prompt`. This is the length of the adaption prompt to insert."
            )

        if not self.adapter_layers:
            raise ConfigValidationError(
                "`adapter_layers` must be set to a value greater than 0 when finetuning is enabled and the adapter"
                "type is `adaption_prompt`. This is the number of adapter layers to insert."
            )

    type: str = schema_utils.ProtectedString(
        "adaption_prompt",
        description=LLM_METADATA["adapter"]["adaption_prompt"]["type"].long_description,
    )

    adapter_len: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of adapter tokens to insert.",
        parameter_metadata=LLM_METADATA["adapter"]["adaption_prompt"]["adapter_len"],
    )

    adapter_layers: int = schema_utils.PositiveInteger(
        default=1,
        allow_none=False,
        description="Number of adapter layers to insert (from the top).",
        parameter_metadata=LLM_METADATA["adapter"]["adaption_prompt"]["adapter_layers"],
    )

    def to_config(self, task_type: str = None, **kwargs) -> "PeftConfig":
        from peft import AdaptionPromptConfig as _AdaptionPromptConfig

        return _AdaptionPromptConfig(
            adapter_len=self.adapter_len,
            adapter_layers=self.adapter_layers,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "Adaption Prompt"

    @classmethod
    def description(cls) -> str:
        return LLM_METADATA["adapter"]["adaption_prompt"]["type"].long_description


@DeveloperAPI
@register_adapter("ia3")
class IA3Config(BaseAdapterConfig):
    type: str = schema_utils.ProtectedString(
        "ia3",
        description=LLM_METADATA["adapter"]["ia3"]["type"].long_description,
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="The names of the modules to apply (IA)^3 to.",
        parameter_metadata=LLM_METADATA["adapter"]["ia3"]["target_modules"],
    )

    feedforward_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description=(
            "The names of the modules to be treated as feedforward modules, as in the original paper. These modules "
            "will have (IA)^3 vectors multiplied to the input, instead of the output. feedforward_modules must be a "
            "name or a subset of names present in target_modules."
        ),
        parameter_metadata=LLM_METADATA["adapter"]["ia3"]["feedforward_modules"],
    )

    fan_in_fan_out: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses "
            "Conv1D which stores weights like (fan_in, fan_out) and hence this should be set to True. "
        ),
        parameter_metadata=LLM_METADATA["adapter"]["ia3"]["fan_in_fan_out"],
    )

    modules_to_save: list[str] | None = schema_utils.List(
        list_type=str,
        default=None,
        allow_none=True,
        description=(
            "List of modules apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint."
        ),
        parameter_metadata=LLM_METADATA["adapter"]["ia3"]["modules_to_save"],
    )

    init_ia3_weights: bool = schema_utils.Boolean(
        default=True,
        description="Whether to initialize the vectors in the (IA)^3 layers, defaults to True.",
        parameter_metadata=LLM_METADATA["adapter"]["ia3"]["init_ia3_weights"],
    )

    def to_config(self, task_type: str = None, **kwargs) -> "PeftConfig":
        from peft import IA3Config as _IA3Config

        return _IA3Config(
            target_modules=self.target_modules,
            feedforward_modules=self.feedforward_modules,
            fan_in_fan_out=self.fan_in_fan_out,
            modules_to_save=self.modules_to_save,
            init_ia3_weights=self.init_ia3_weights,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "IA3"

    @classmethod
    def description(cls) -> str:
        return LLM_METADATA["adapter"]["ia3"]["type"].long_description


@DeveloperAPI
@register_adapter(name="vera")
class VeraAdapterConfig(BaseAdapterConfig):
    """VeRA: Vector-based Random Matrix Adaptation (ICLR 2024).

    Uses shared frozen random matrices with trained scaling vectors. 10x fewer trainable
    parameters than LoRA, useful for extreme parameter efficiency and multi-tenant serving.
    """

    type: str = schema_utils.ProtectedString("vera", description="VeRA adapter.")

    r: int = schema_utils.PositiveInteger(default=256, description="VeRA rank dimension.")

    target_modules: list[str] | None = schema_utils.List(
        default=None, allow_none=True, description="List of module names to apply VeRA to."
    )

    projection_prng_key: int = schema_utils.NonNegativeInteger(
        default=0, description="PRNG key for shared random projection matrices."
    )

    def to_config(self, task_type: str = None, **kwargs):
        from peft import VeraConfig as _VeraConfig

        return _VeraConfig(
            r=self.r,
            target_modules=self.target_modules,
            projection_prng_key=self.projection_prng_key,
            task_type=task_type,
        )


@DeveloperAPI
@register_adapter(name="loha")
class LoHaAdapterConfig(BaseAdapterConfig):
    """LoHa: Low-Rank Hadamard Product Adaptation.

    Uses Hadamard product of two low-rank matrices for parameter-efficient fine-tuning.
    Can capture more complex weight updates than LoRA at the same rank.
    """

    type: str = schema_utils.ProtectedString("loha", description="LoHa adapter.")

    r: int = schema_utils.PositiveInteger(default=8, description="LoHa rank dimension.")

    alpha: float = schema_utils.Float(default=8, description="Scaling factor for LoHa.")

    target_modules: list[str] | None = schema_utils.List(
        default=None, allow_none=True, description="List of module names to apply LoHa to."
    )

    def to_config(self, task_type: str = None, **kwargs):
        from peft import LoHaConfig as _LoHaConfig

        return _LoHaConfig(r=self.r, alpha=self.alpha, target_modules=self.target_modules, task_type=task_type)


@DeveloperAPI
@register_adapter(name="lokr")
class LoKrAdapterConfig(BaseAdapterConfig):
    """LoKr: Low-Rank Kronecker Product Adaptation.

    Uses Kronecker product decomposition for efficient weight updates.
    """

    type: str = schema_utils.ProtectedString("lokr", description="LoKr adapter.")

    r: int = schema_utils.PositiveInteger(default=8, description="LoKr rank dimension.")

    alpha: float = schema_utils.Float(default=8, description="Scaling factor for LoKr.")

    target_modules: list[str] | None = schema_utils.List(
        default=None, allow_none=True, description="List of module names to apply LoKr to."
    )

    def to_config(self, task_type: str = None, **kwargs):
        from peft import LoKrConfig as _LoKrConfig

        return _LoKrConfig(r=self.r, alpha=self.alpha, target_modules=self.target_modules, task_type=task_type)


@DeveloperAPI
@register_adapter(name="fourierft")
class FourierFTAdapterConfig(BaseAdapterConfig):
    """FourierFT: Frequency-domain fine-tuning.

    Learns weight updates in the Fourier frequency domain, providing a different
    inductive bias than spatial methods like LoRA.
    """

    type: str = schema_utils.ProtectedString("fourierft", description="FourierFT adapter.")

    n_frequency: int = schema_utils.PositiveInteger(default=1000, description="Number of frequency components.")

    scaling: float = schema_utils.Float(default=150.0, description="Scaling factor for FourierFT.")

    target_modules: list[str] | None = schema_utils.List(
        default=None, allow_none=True, description="List of module names to apply FourierFT to."
    )

    def to_config(self, task_type: str = None, **kwargs):
        from peft import FourierFTConfig as _FourierFTConfig

        return _FourierFTConfig(
            n_frequency=self.n_frequency,
            scaling=self.scaling,
            target_modules=self.target_modules,
            task_type=task_type,
        )


@DeveloperAPI
@register_adapter(name="boft")
class BOFTAdapterConfig(BaseAdapterConfig):
    """BOFT: Butterfly Orthogonal Fine-Tuning.

    Uses butterfly factorization to learn orthogonal transformations, preserving
    the pre-trained model's hyperspherical energy while adapting to new tasks.
    """

    type: str = schema_utils.ProtectedString("boft", description="BOFT adapter.")

    boft_block_size: int = schema_utils.PositiveInteger(
        default=4, description="Block size for butterfly factorization."
    )

    boft_n_butterfly_factor: int = schema_utils.PositiveInteger(default=1, description="Number of butterfly factors.")

    boft_dropout: float = schema_utils.NonNegativeFloat(default=0.05, description="Dropout for BOFT layers.")

    target_modules: list[str] | None = schema_utils.List(
        default=None, allow_none=True, description="List of module names to apply BOFT to."
    )

    def to_config(self, task_type: str = None, **kwargs):
        from peft import BOFTConfig as _BOFTConfig

        return _BOFTConfig(
            boft_block_size=self.boft_block_size,
            boft_n_butterfly_factor=self.boft_n_butterfly_factor,
            boft_dropout=self.boft_dropout,
            target_modules=self.target_modules,
            task_type=task_type,
        )


@DeveloperAPI
def get_adapter_conds():
    conds = []
    for adapter_type, adapter_cls in adapter_registry.items():
        other_props = schema_utils.unload_jsonschema_from_config_class(adapter_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": adapter_type},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def AdapterDataclassField(default: str | None = None):
    description = "Whether to use parameter-efficient fine-tuning"

    class AdapterSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(
                registry=adapter_registry,
                default_value=default,
                description=description,
                parameter_metadata=None,
                allow_str_value=True,
                allow_none=True,
            )

        def get_schema_from_registry(self, key: str) -> type[schema_utils.LudwigBaseConfig]:
            return adapter_registry[key]

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": list(adapter_registry.keys()),
                                "description": "The type of PEFT adapter to use during fine-tuning",
                            },
                        },
                        "title": "Perform parameter efficient fine-tuning",
                        "allOf": get_adapter_conds(),
                        "required": ["type"],
                        "description": "The type of PEFT adapter to use during fine-tuning",
                        "parameter_metadata": convert_metadata_to_json(LLM_METADATA["adapter"]["_oneOf"]["allOf"]),
                    },
                    {
                        "type": "null",
                        "title": "adapter_null_option",
                        "description": "Disable the adapter.",
                        "parameter_metadata": convert_metadata_to_json(LLM_METADATA["adapter"]["_oneOf"]["none"]),
                    },
                ],
                "title": "adapter_options",
                "description": "Whether to use parameter-efficient fine-tuning",
                "parameter_metadata": convert_metadata_to_json(LLM_METADATA["adapter"]["_meta"]),
                "default": default,
            }

    return AdapterSelection().get_default_field()


# ================================================================================================
# Multi-adapter support
# ================================================================================================
#
# The singular `adapter:` field above supports the common case: one adapter attached to one
# base model. The `adapters:` field below supports the multi-adapter case: several named
# adapters attached to the same base model, switchable at runtime via PEFT's `set_adapter()`
# and optionally merged via `add_weighted_adapter()` using combination types like TIES and
# DARE (Yadav et al., NeurIPS 2023 / Yu et al., ICML 2024).
#
# `adapter:` and `adapters:` are mutually exclusive — a config must use one form or the other.
# Back-compat: existing configs that set `adapter:` continue to work unchanged.


@DeveloperAPI
class MergeAdaptersConfig(schema_utils.LudwigBaseConfig):
    """Optional weighted merge over a subset of the named adapters.

    Produces a new adapter registered under ``name`` by combining ``sources`` with the
    matching ``weights`` under ``combination_type``. The merged adapter is added to the
    model alongside the sources; pick it as ``active`` to make it the default at
    inference time.
    """

    name: str = schema_utils.String(
        default="merged",
        description="Name to register the merged adapter under.",
    )

    sources: list | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="Names of the adapters to merge. Each name must appear in the `adapters` map.",
    )

    weights: list | None = schema_utils.List(
        default=None,
        allow_none=True,
        description=(
            "Per-source weights; must have the same length as `sources`. " "If null, all weights default to 1.0."
        ),
    )

    combination_type: str = schema_utils.StringOptions(
        options=["linear", "svd", "ties", "dare_linear", "dare_ties", "magnitude_prune"],
        default="linear",
        allow_none=False,
        description=(
            "PEFT weighted-merge combination type. 'linear' is a plain weighted sum. "
            "'ties' (Yadav et al., NeurIPS 2023) resolves sign conflicts across source "
            "deltas before merging. 'dare_linear' / 'dare_ties' (Yu et al., ICML 2024) "
            "prune a fraction `density` of deltas before merging for smaller footprints."
        ),
    )

    density: float = schema_utils.FloatRange(
        default=0.5,
        min=0.0,
        max=1.0,
        description=(
            "Fraction of weight deltas kept when `combination_type` is 'ties', "
            "'dare_linear', 'dare_ties', or 'magnitude_prune'. Ignored for 'linear' / 'svd'."
        ),
    )


@DeveloperAPI
class MergeAdaptersConfigField(schema_utils.NestedConfigField):
    def __init__(self):
        super().__init__(MergeAdaptersConfig, allow_none=True, default_missing=True)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_config_class(MergeAdaptersConfig, title="MergeAdapters")


@DeveloperAPI
class NamedAdaptersConfig(schema_utils.LudwigBaseConfig):
    """Configuration for multiple named PEFT adapters on the same base model."""

    adapters: dict | None = schema_utils.Dict(
        default=None,
        allow_none=False,
        description=(
            "Mapping of adapter name -> adapter config. Each value is a regular adapter "
            "config (e.g. ``{type: lora, r: 8}``) identical to what the singular "
            "`adapter:` field accepts. Adapter names must be unique; PEFT will register "
            "each one on the model and the first-listed adapter becomes the active one "
            "unless `active` is set."
        ),
    )

    active: str | None = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "Name of the adapter to activate after all adapters are registered. "
            "If null, the first entry in `adapters` is used. Set this to a merged adapter "
            "name from `merge:` to activate the merged adapter at inference time."
        ),
    )

    merge: MergeAdaptersConfig | None = MergeAdaptersConfigField().get_default_field()


@DeveloperAPI
class NamedAdaptersConfigField(schema_utils.NestedConfigField):
    def __init__(self):
        super().__init__(NamedAdaptersConfig, allow_none=True, default_missing=True)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_config_class(NamedAdaptersConfig, title="NamedAdapters")


@DeveloperAPI
def NamedAdaptersDataclassField():
    return NamedAdaptersConfigField().get_default_field()
