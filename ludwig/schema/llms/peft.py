from abc import ABC, abstractmethod
from typing import Optional, Type, TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.schema.utils import ludwig_dataclass
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
@ludwig_dataclass
class LoraPostprocessorConfig(schema_utils.BaseMarshmallowConfig):
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
class LoraPostprocessorConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(LoraPostprocessorConfig)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(LoraPostprocessorConfig, title="LoraPostprocessor")


@DeveloperAPI
@ludwig_dataclass
class BaseAdapterConfig(schema_utils.BaseMarshmallowConfig, ABC):
    type: str

    pretrained_adapter_weights: Optional[str] = schema_utils.String(
        default=None, description="Path to pretrained weights.", allow_none=True
    )

    postprocessor: LoraPostprocessorConfig = LoraPostprocessorConfigField().get_default_field()

    @abstractmethod
    def to_config(self, **kwargs) -> "PeftConfig":
        pass


@DeveloperAPI
@register_adapter(name="lora")
@ludwig_dataclass
class LoraConfig(BaseAdapterConfig):
    type: str = schema_utils.ProtectedString(
        "lora",
        description=LLM_METADATA["adapter"]["lora"]["type"].long_description,
    )

    r: int = schema_utils.PositiveInteger(
        default=8,
        description="Lora attention dimension.",
        parameter_metadata=LLM_METADATA["adapter"]["lora"]["r"],
    )

    alpha: int = schema_utils.PositiveInteger(
        default=16,
        description="The alpha parameter for Lora scaling.",
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

    def to_config(self, task_type: str = None, **kwargs) -> "PeftConfig":
        from peft import LoraConfig as _LoraConfig

        return _LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias_type,
            task_type=task_type,
        )

    @classmethod
    def name(cls) -> str:
        return "LoRA"

    @classmethod
    def description(cls) -> str:
        return LLM_METADATA["adapter"]["lora"]["type"].long_description


@DeveloperAPI
@ludwig_dataclass
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

    token_dim: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The hidden embedding dimension of the base transformer model.",
    )

    num_transformer_submodules: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of transformer submodules in the base transformer model.",
    )

    num_attention_heads: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of attention heads in the base transformer model.",
    )

    num_layers: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The number of layers in the base transformer model.",
    )


# TODO(travis): fix text generation when using prompt tuning:
#     RuntimeError: shape '[-1, 17]' is invalid for input of size 9
# @DeveloperAPI
# @register_adapter("prompt_tuning")
# @ludwig_dataclass
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


# TODO(travis): fix prefix tuning and p-tuning to work with DDP
# @DeveloperAPI
# @register_adapter("prefix_tuning")
# @schema_utils.ludwig_dataclass
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
# @ludwig_dataclass
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
@ludwig_dataclass
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

    total_step: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The total training steps that should be specified before training.",
    )

    rank_pattern: Optional[dict] = schema_utils.Dict(
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
@register_adapter("adaption_prompt")
@ludwig_dataclass
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
def get_adapter_conds():
    conds = []
    for adapter_type, adapter_cls in adapter_registry.items():
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(adapter_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": adapter_type},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def AdapterDataclassField(default: Optional[str] = None):
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

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
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
