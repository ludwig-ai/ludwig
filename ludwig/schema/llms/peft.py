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
class EvaSubConfig(schema_utils.LudwigBaseConfig):
    """Configuration for EVA (Explained Variance Adaptation) LoRA initialization.

    EVA initializes LoRA based on the SVD of layer input activations, achieving state-of-the-art
    performance by adapting the adapter directions to the actual data distribution.
    Paper: https://arxiv.org/abs/2410.07170
    """

    rho: float = schema_utils.NonNegativeFloat(
        default=2.0,
        description="Scaling factor for EVA. Controls how strongly activations influence the initialization.",
    )
    tau: float = schema_utils.FloatRange(
        default=0.99,
        min=0.0,
        max=1.0,
        description="Momentum for running statistics in EVA.",
    )
    use_label_mask: bool = schema_utils.Boolean(
        default=True,
        description="Whether to mask padding/ignore tokens when computing activation statistics.",
    )
    label_mask_value: int = schema_utils.Integer(
        default=-100,
        description="Token id to mask out (usually the ignore_index for cross-entropy loss).",
    )
    whiten: bool = schema_utils.Boolean(
        default=False,
        description="Whether to whiten activations before computing SVD.",
    )
    adjust_scaling_factors: bool = schema_utils.Boolean(
        default=True,
        description="Adjust LoRA scaling factors after EVA initialization to preserve pre-trained model outputs.",
    )


@DeveloperAPI
class EvaSubConfigField(schema_utils.NestedConfigField):
    def __init__(self):
        super().__init__(EvaSubConfig, allow_none=True, default_missing=True)

    def _jsonschema_type_mapping(self):
        inner = schema_utils.unload_jsonschema_from_config_class(EvaSubConfig, title="EvaConfig")
        return {"oneOf": [inner, {"type": "null"}]}


@DeveloperAPI
class LoftQSubConfig(schema_utils.LudwigBaseConfig):
    """Configuration for LoftQ quantization-aware LoRA initialization.

    LoftQ simultaneously quantizes the backbone weights and initializes LoRA adapters
    to minimize the quantization error. Requires `init_lora_weights='loftq'`.
    Paper: https://arxiv.org/abs/2310.08659
    """

    loftq_bits: int = schema_utils.IntegerOptions(
        options=[2, 4, 8],
        default=4,
        description="Number of bits for LoftQ quantization (2, 4, or 8).",
    )
    loftq_iter: int = schema_utils.PositiveInteger(
        default=1,
        description="Number of LoftQ iterations. More iterations improve approximation quality.",
    )


@DeveloperAPI
class LoftQSubConfigField(schema_utils.NestedConfigField):
    def __init__(self):
        super().__init__(LoftQSubConfig, allow_none=True, default_missing=True)

    def _jsonschema_type_mapping(self):
        inner = schema_utils.unload_jsonschema_from_config_class(LoftQSubConfig, title="LoftQConfig")
        return {"oneOf": [inner, {"type": "null"}]}


@DeveloperAPI
@register_adapter(name="lora")
class LoraConfig(BaseAdapterConfig):
    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.r * 2
        if self.init_lora_weights == "loftq" and self.loftq_config is None:
            raise ConfigValidationError(
                "`loftq_config` must be set when `init_lora_weights` is 'loftq'. "
                "Example: loftq_config: {loftq_bits: 4, loftq_iter: 1}"
            )
        if self.init_lora_weights == "eva" and self.eva_config is None:
            raise ConfigValidationError(
                "`eva_config` must be set when `init_lora_weights` is 'eva'. Example: eva_config: {rho: 2.0}"
            )

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

    init_lora_weights: str | bool = schema_utils.StringOptions(
        options=["default", "gaussian", "eva", "olora", "pissa", "corda", "loftq", "orthogonal"],
        default="default",
        allow_none=False,
        description=(
            "Initialization strategy for LoRA weight matrices. "
            "'default' uses the standard Kaiming uniform init (A) and zeros (B). "
            "'gaussian' uses Gaussian init for A. "
            "'pissa' (Principal Singular values and Singular vectors Adaptation) initializes using SVD of the "
            "pretrained weight, converging faster and often outperforming standard LoRA. "
            "Paper: https://arxiv.org/abs/2404.02948. "
            "'eva' (Explained Variance Adaptation) initializes from the SVD of layer input activations — "
            "requires `eva_config` to be set. Paper: https://arxiv.org/abs/2410.07170. "
            "'corda' (Context-Oriented Decomposition Adaptation) combines PiSSA and full fine-tuning signals, "
            "converging faster than PiSSA. Paper: https://arxiv.org/abs/2406.05223. "
            "'olora' (Orthonormal LoRA) uses QR decomposition for better conditioning. "
            "'loftq' (LoftQ) jointly quantizes base weights and initializes LoRA — "
            "requires `loftq_config` to be set. Paper: https://arxiv.org/abs/2310.08659. "
            "'orthogonal' uses orthogonal initialization."
        ),
    )

    eva_config: EvaSubConfig | None = EvaSubConfigField().get_default_field()

    loftq_config: LoftQSubConfig | None = LoftQSubConfigField().get_default_field()

    rank_pattern: dict | None = schema_utils.Dict(
        default=None,
        allow_none=True,
        description=(
            "Per-layer rank overrides as a mapping of layer name (or regex) to rank integer. "
            "Overrides the global `r` for matched layers. Useful for LoRA-XS style configurations "
            "where different layers benefit from different ranks. "
            "Example: {'model.layers.0.self_attn.q_proj': 4, 'model.layers.0.self_attn.v_proj': 2}"
        ),
    )

    alpha_pattern: dict | None = schema_utils.Dict(
        default=None,
        allow_none=True,
        description=(
            "Per-layer alpha (scaling) overrides as a mapping of layer name (or regex) to float. "
            "Overrides the global `alpha` for matched layers."
        ),
    )

    layer_replication: list | None = schema_utils.List(
        default=None,
        allow_none=True,
        description=(
            "Layer replication configuration as a list of [start, end] index pairs. Enables depth-wise "
            "parameter efficiency by sharing LoRA weights across layer ranges. "
            "Example: [[0, 4], [2, 5]] creates two overlapping groups."
        ),
    )

    def to_config(self, task_type: str | None = None, **kwargs) -> "PeftConfig":
        from peft import LoraConfig as _LoraConfig

        init_weights = self.init_lora_weights
        if init_weights == "default":
            init_weights = True

        eva_config = None
        loftq_config = None
        if init_weights == "eva" and self.eva_config is not None:
            from peft import EvaConfig as _EvaConfig

            eva_config = _EvaConfig(
                rho=self.eva_config.rho,
                tau=self.eva_config.tau,
                use_label_mask=self.eva_config.use_label_mask,
                label_mask_value=self.eva_config.label_mask_value,
                whiten=self.eva_config.whiten,
                adjust_scaling_factors=self.eva_config.adjust_scaling_factors,
            )
        elif init_weights == "loftq" and self.loftq_config is not None:
            from peft import LoftQConfig as _LoftQConfig

            loftq_config = _LoftQConfig(
                loftq_bits=self.loftq_config.loftq_bits,
                loftq_iter=self.loftq_config.loftq_iter,
            )

        layer_replication = None
        if self.layer_replication is not None:
            layer_replication = [tuple(pair) for pair in self.layer_replication]

        return _LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias_type,
            target_modules=self.target_modules,
            task_type=task_type,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
            init_lora_weights=init_weights,
            eva_config=eva_config,
            loftq_config=loftq_config,
            rank_pattern=self.rank_pattern or {},
            alpha_pattern=self.alpha_pattern or {},
            layer_replication=layer_replication,
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

    def to_config(self, task_type: str | None = None, **kwargs) -> "PeftConfig":
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

    def to_config(self, task_type: str | None = None, **kwargs) -> "PeftConfig":
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

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import VeraConfig as _VeraConfig

        return _VeraConfig(
            r=self.r,
            target_modules=self.target_modules,
            projection_prng_key=self.projection_prng_key,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "VeRA"

    @classmethod
    def description(cls) -> str:
        return "Vector-based Random Matrix Adaptation. Shares frozen random matrices across layers; only small scaling vectors are trained, giving 10× fewer parameters than LoRA at the same rank."


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

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import LoHaConfig as _LoHaConfig

        return _LoHaConfig(r=self.r, alpha=self.alpha, target_modules=self.target_modules, task_type=task_type)

    @classmethod
    def name(cls) -> str:
        return "LoHa"

    @classmethod
    def description(cls) -> str:
        return "Low-Rank Hadamard Product Adaptation. Uses a Hadamard product of two low-rank matrices to capture more complex weight updates than LoRA at the same rank."


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

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import LoKrConfig as _LoKrConfig

        return _LoKrConfig(r=self.r, alpha=self.alpha, target_modules=self.target_modules, task_type=task_type)

    @classmethod
    def name(cls) -> str:
        return "LoKr"

    @classmethod
    def description(cls) -> str:
        return "Low-Rank Kronecker Product Adaptation. Uses Kronecker product decomposition for efficient weight updates with a different inductive bias than LoRA."


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

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import FourierFTConfig as _FourierFTConfig

        return _FourierFTConfig(
            n_frequency=self.n_frequency,
            scaling=self.scaling,
            target_modules=self.target_modules,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "FourierFT"

    @classmethod
    def description(cls) -> str:
        return "Frequency-domain fine-tuning. Learns weight updates in the Fourier frequency domain, providing a complementary inductive bias to spatial methods like LoRA."


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

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import BOFTConfig as _BOFTConfig

        return _BOFTConfig(
            boft_block_size=self.boft_block_size,
            boft_n_butterfly_factor=self.boft_n_butterfly_factor,
            boft_dropout=self.boft_dropout,
            target_modules=self.target_modules,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "BOFT"

    @classmethod
    def description(cls) -> str:
        return "Butterfly Orthogonal Fine-Tuning. Learns orthogonal transformations via butterfly factorization, preserving the pre-trained model's geometry while adapting to new tasks."


@DeveloperAPI
@register_adapter(name="tinylora")
class TinyLoraAdapterConfig(BaseAdapterConfig):
    """TinyLoRA: Extreme parameter-efficient fine-tuning via SVD projection.

    Uses SVD decomposition of frozen weights and projects a tiny trainable vector through fixed random tensors.
    Enables fine-tuning in as few as 13 parameters. Ideal for extremely constrained hardware or edge deployment.
    Paper: https://arxiv.org/abs/2602.04118
    """

    type: str = schema_utils.ProtectedString("tinylora", description="TinyLoRA adapter (LoRA-XS equivalent).")

    r: int = schema_utils.PositiveInteger(
        default=2,
        description="SVD rank for the frozen U, Sigma, V decomposition. The paper recommends r=2.",
    )

    u: int = schema_utils.PositiveInteger(
        default=64,
        description=(
            "Trainable vector dimension per group. Controls the expressivity of the adaptation. "
            "Can be as low as 1–13 for extreme parameter efficiency."
        ),
    )

    weight_tying: float = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        description=(
            "Degree of weight tying across target modules (0.0 = no sharing, 1.0 = full sharing). "
            "Sharing trainable vectors across modules further reduces parameter count."
        ),
    )

    projection_seed: int = schema_utils.NonNegativeInteger(
        default=42,
        description="Random seed for generating the fixed projection matrices.",
    )

    save_projection: bool = schema_utils.Boolean(
        default=True,
        description="Whether to save the projection tensors in the state dict.",
    )

    tinylora_dropout: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Dropout probability for TinyLoRA layers.",
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of module names or regex to apply TinyLoRA to.",
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import TinyLoraConfig as _TinyLoraConfig

        return _TinyLoraConfig(
            r=self.r,
            u=self.u,
            weight_tying=self.weight_tying,
            projection_seed=self.projection_seed,
            save_projection=self.save_projection,
            tinylora_dropout=self.tinylora_dropout,
            target_modules=self.target_modules,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "TinyLoRA"

    @classmethod
    def description(cls) -> str:
        return "TinyLoRA: extreme parameter-efficient fine-tuning via SVD projection (LoRA-XS variant)."


@DeveloperAPI
@register_adapter(name="c3a")
class C3AAdapterConfig(BaseAdapterConfig):
    """C3A: Contextual / Conditional / Compositional Adapter.

    Uses block-diagonal matrices for structured parameter efficiency.
    Enables context-aware adapter routing and multi-task modularity.
    """

    type: str = schema_utils.ProtectedString("c3a", description="C3A adapter.")

    block_size: int = schema_utils.PositiveInteger(
        default=256,
        description=(
            "Block size for C3A, must be divisible by both the input size and the output size of each target layer. "
            "Setting this to the GCD of all target layer dimensions is a safe default. "
            "Larger block sizes mean fewer parameters."
        ),
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of module names or regex to apply C3A to.",
    )

    bias_type: str = schema_utils.StringOptions(
        options=["none", "all", "c3a_only"],
        default="none",
        description="Bias type for C3A. 'none' trains no biases; 'all' or 'c3a_only' trains the adapter biases.",
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import C3AConfig as _C3AConfig

        return _C3AConfig(
            block_size=self.block_size,
            target_modules=self.target_modules,
            bias=self.bias_type,
            block_size_pattern={},
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "C3A"

    @classmethod
    def description(cls) -> str:
        return "C3A: context-aware block-diagonal adapter for multi-task and compositional fine-tuning."


@DeveloperAPI
@register_adapter(name="oft")
class OFTAdapterConfig(BaseAdapterConfig):
    """OFT: Orthogonal Fine-Tuning.

    Applies orthogonal transformations to the weight matrices, preserving the hyperspherical energy
    of the pre-trained model while adapting to new tasks. Particularly effective for maintaining
    output diversity and preventing catastrophic forgetting.
    Paper: https://arxiv.org/abs/2306.07280
    """

    type: str = schema_utils.ProtectedString("oft", description="OFT adapter.")

    r: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "OFT rank. When 0, the block size (`oft_block_size`) controls the granularity instead. "
            "Cannot be set simultaneously with `oft_block_size`."
        ),
    )

    oft_block_size: int = schema_utils.PositiveInteger(
        default=32,
        description="Block size for the butterfly factorization of the orthogonal transform.",
    )

    module_dropout: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Probability of randomly zeroing an OFT block during training.",
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of module names or regex to apply OFT to.",
    )

    coft: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use Constrained OFT (COFT), which enforces the constraint ||I - R^T R||_F <= eps.",
    )

    eps: float = schema_utils.NonNegativeFloat(
        default=6e-5,
        description="Constraint strength for COFT (only used when `coft=True`).",
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import OFTConfig as _OFTConfig

        return _OFTConfig(
            r=self.r,
            oft_block_size=self.oft_block_size,
            module_dropout=self.module_dropout,
            target_modules=self.target_modules,
            coft=self.coft,
            eps=self.eps,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "OFT"

    @classmethod
    def description(cls) -> str:
        return "OFT: Orthogonal Fine-Tuning that preserves hyperspherical energy of the pre-trained model."


@DeveloperAPI
@register_adapter(name="hra")
class HRAAdapterConfig(BaseAdapterConfig):
    """HRA: Householder Reflection Adaptation.

    Parameterizes weight updates as products of Householder reflections, which are orthogonal by construction.
    Provides stronger expressivity than OFT with fewer hyperparameters.
    Paper: https://arxiv.org/abs/2405.17484
    """

    type: str = schema_utils.ProtectedString("hra", description="HRA adapter.")

    r: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of Householder reflections (rank). More reflections = more expressive adaptation.",
    )

    apply_GS: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Whether to apply Gram-Schmidt orthogonalization to the Householder vectors. "
            "Improves numerical stability at the cost of a small overhead."
        ),
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of module names or regex to apply HRA to.",
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import HRAConfig as _HRAConfig

        return _HRAConfig(
            r=self.r,
            apply_GS=self.apply_GS,
            target_modules=self.target_modules,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "HRA"

    @classmethod
    def description(cls) -> str:
        return "HRA: Householder Reflection Adaptation — orthogonal updates via Householder reflections."


@DeveloperAPI
@register_adapter(name="waveft")
class WaveFTAdapterConfig(BaseAdapterConfig):
    """WaveFT: Wavelet-domain Fine-Tuning.

    Learns weight updates in the wavelet frequency domain using discrete wavelet transforms.
    Provides a different inductive bias from spatial methods like LoRA, often benefiting
    tasks with structured or periodic patterns.
    Paper: https://arxiv.org/abs/2411.09295
    """

    type: str = schema_utils.ProtectedString("waveft", description="WaveFT adapter.")

    n_frequency: int = schema_utils.PositiveInteger(
        default=2592,
        description="Number of wavelet frequency components to learn. Fewer = more parameter efficient.",
    )

    scaling: float = schema_utils.NonNegativeFloat(
        default=25.0,
        description="Scaling factor applied to the wavelet-domain updates.",
    )

    wavelet_family: str = schema_utils.StringOptions(
        options=["db1", "db2", "db3", "haar", "sym2", "coif1"],
        default="db1",
        description=(
            "Wavelet family to use for the discrete wavelet transform. "
            "'db1'/'haar' are simplest; higher-order Daubechies ('db2', 'db3') capture smoother features."
        ),
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of module names or regex to apply WaveFT to.",
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import WaveFTConfig as _WaveFTConfig

        return _WaveFTConfig(
            n_frequency=self.n_frequency,
            scaling=self.scaling,
            wavelet_family=self.wavelet_family,
            target_modules=self.target_modules,
            n_frequency_pattern={},
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "WaveFT"

    @classmethod
    def description(cls) -> str:
        return "WaveFT: Wavelet-domain fine-tuning with structured frequency-domain weight updates."


@DeveloperAPI
@register_adapter(name="ln_tuning")
class LNTuningAdapterConfig(BaseAdapterConfig):
    """LN-Tuning: Layer Normalization Tuning.

    Fine-tunes only the layer normalization parameters (weight and bias) of the model.
    Extremely parameter-efficient — often only ~0.1% of total parameters — while surprisingly
    effective for domain adaptation tasks.
    """

    type: str = schema_utils.ProtectedString("ln_tuning", description="LN-Tuning adapter.")

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description=(
            "List of layer norm module names or regex to tune. "
            "Defaults to all LayerNorm / RMSNorm modules in the model."
        ),
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import LNTuningConfig as _LNTuningConfig

        return _LNTuningConfig(
            target_modules=self.target_modules,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "LN-Tuning"

    @classmethod
    def description(cls) -> str:
        return "LN-Tuning: tunes only the layer normalization parameters for ultra-lightweight adaptation."


@DeveloperAPI
@register_adapter(name="vblora")
class VBLoRAAdapterConfig(BaseAdapterConfig):
    """VBLoRA: Vector Bank LoRA.

    Represents LoRA weight matrices as a sparse combination of shared vectors from a global bank.
    Achieves significant parameter compression by reusing vectors across layers.
    Paper: https://arxiv.org/abs/2405.15179
    """

    type: str = schema_utils.ProtectedString("vblora", description="VBLoRA adapter.")

    r: int = schema_utils.PositiveInteger(
        default=4,
        description="LoRA rank dimension. Controls the bottleneck size of each adaptation.",
    )

    num_vectors: int = schema_utils.PositiveInteger(
        default=256,
        description="Number of vectors in the global vector bank shared across all layers.",
    )

    vector_length: int = schema_utils.PositiveInteger(
        default=256,
        description="Length (dimension) of each vector in the bank. Usually set to the hidden size or head dim.",
    )

    topk: int = schema_utils.PositiveInteger(
        default=2,
        description=(
            "Number of top-k vectors selected from the bank for each LoRA matrix column. "
            "Higher k increases expressivity but also parameter count."
        ),
    )

    vblora_dropout: float = schema_utils.NonNegativeFloat(
        default=0.0,
        description="Dropout probability for VBLoRA layers.",
    )

    save_only_topk_weights: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save only the top-k selection logits rather than the full bank weights.",
    )

    target_modules: list[str] | None = schema_utils.List(
        default=None,
        allow_none=True,
        description="List of module names or regex to apply VBLoRA to.",
    )

    def to_config(self, task_type: str | None = None, **kwargs):
        from peft import VBLoRAConfig as _VBLoRAConfig

        return _VBLoRAConfig(
            r=self.r,
            num_vectors=self.num_vectors,
            vector_length=self.vector_length,
            topk=self.topk,
            vblora_dropout=self.vblora_dropout,
            save_only_topk_weights=self.save_only_topk_weights,
            target_modules=self.target_modules,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "VBLoRA"

    @classmethod
    def description(cls) -> str:
        return "VBLoRA: Vector Bank LoRA that shares vectors across layers for extreme compression."


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
            "Per-source weights; must have the same length as `sources`. If null, all weights default to 1.0."
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
