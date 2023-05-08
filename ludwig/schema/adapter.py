from abc import ABC
from typing import List, Optional, Type, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

_adapter_registry = Registry()


@DeveloperAPI
def register_adapter(name: str):
    """Registers an adapter config class with the adapter config registry."""

    def wrap(adapter_config: Union[BasePeftConfig, BasePromptLearningConfig]):
        _adapter_registry[name] = adapter_config
        return adapter_config

    return wrap


@DeveloperAPI
def get_adapter_cls(name: str):
    """Returns the adapter config class registered with the given name."""
    return _adapter_registry[name]


@DeveloperAPI
@schema_utils.ludwig_dataclass
class BasePeftConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Config for prompt learning adapters. Not meant to be used directly.

    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/utils/config.py (PeftConfig)
    """

    peft_type: str

    # Hard coded since we default to using AutoModelForCausalLM for LLM model types.
    # TODO(Arnav): Remove this restriction when we support other AutoModel types like Seq2SeqLM, etc.
    task_type: str = schema_utils.ProtectedString("CAUSAL_LM")

    inference_mode: bool = schema_utils.Boolean(
        default=False,
        allow_none=True,
        description="Whether to use the model in inference mode. In inference mode, the model will not be trained.",
    )


@DeveloperAPI
@schema_utils.ludwig_dataclass
class BasePromptLearningConfig(BasePeftConfig):
    """Config for prompt learning adapters. Not meant to be used directly.

    Adapted from https://github.com/huggingface/peft/blob/main/src/peft/utils/config.py (PromptLearningConfig)
    """

    num_virtual_tokens: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="Number of virtual tokens to add to the prompt. Virtual tokens are used to control the behavior of "
        " the model during inference. ",
    )

    token_dim: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The hidden embedding dimension of the base transformer model.",
    )

    num_transformer_submodules: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The number of transformer submodules in the base transformer model.",
    )

    num_attention_heads: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The number of attention heads in the base transformer model.",
    )

    num_layers: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The number of layers in the base transformer model.",
    )


@DeveloperAPI
@register_adapter("prompt_tuning")
@schema_utils.ludwig_dataclass
class PromptTuningAdapterConfig(BasePromptLearningConfig):
    """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/prompt_tuning.py."""

    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("prompt_tuning")

    peft_type: str = schema_utils.ProtectedString("PROMPT_TUNING")

    # TODO(Arnav): Refactor to allow both RANDOM and TEXT strategies
    prompt_tuning_init: str = schema_utils.ProtectedString(
        "TEXT",
        description="The type of initialization to use for the prompt embedding. ",
    )

    prompt_tuning_init_text: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The text to use to initialize the prompt embedding.",
    )


@DeveloperAPI
@register_adapter("prefix_tuning")
@schema_utils.ludwig_dataclass
class PrefixTuningAdapterconfig(BasePromptLearningConfig):
    """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/prefix_tuning.py."""

    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("prefix_tuning")

    peft_type: str = schema_utils.ProtectedString("PREFIX_TUNING")

    encoder_hidden_size: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The hidden embedding dimension of the prompt encoder.",
    )

    prefix_projection: bool = schema_utils.Boolean(
        default=False,
        allow_none=True,
        description="Whether to use a projection layer in the prompt encoder to project the prefix tokens",
    )


@DeveloperAPI
@register_adapter("p_tuning")
@schema_utils.ludwig_dataclass
class PTuningAdapterConfig(BasePromptLearningConfig):
    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("p_tuning")

    peft_type: str = schema_utils.ProtectedString("P_TUNING")

    encoder_reparameterization_type: str = schema_utils.StringOptions(
        ["MLP", "LSTM"],
        default="MLP",
        allow_none=True,
        description="The type of reparameterization to use for the prompt encoder.",
    )

    encoder_hidden_size: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The hidden embedding dimension of the prompt encoder.",
    )

    encoder_num_layers: int = schema_utils.Integer(
        default=2,
        allow_none=True,
        description="The number of layers in the prompt encoder.",
    )

    encoder_dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="The dropout probability for the prompt encoder.",
    )


@DeveloperAPI
@register_adapter("lora")
@schema_utils.ludwig_dataclass
class LoRAAdapterConfig(BasePeftConfig):
    """Adapted From https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py."""

    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("lora")

    peft_type: str = schema_utils.ProtectedString("LORA")

    r: int = schema_utils.Integer(
        default=8,
        allow_none=True,
        description="LoRA attention dimension",
    )

    # TODO(Arnav): Extended to support regex expression of the module names to replace.
    target_modules: List[str] = schema_utils.List(
        list_type=str,
        default=None,
        allow_none=True,
        description="List of module names to replace with LoRA attention.",
    )

    lora_alpha: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="LoRA alpha parameter",
    )

    lora_dropout: float = schema_utils.FloatRange(
        default=None,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="LoRA dropout probability",
    )

    fan_in_fan_out: bool = schema_utils.Boolean(
        default=False,
        allow_none=True,
        description="Whether to use fan-in/fan-out initialization for LoRA attention."
        "Set this to True if the layer to replace stores weight like (fan_in, fan_out)",
    )

    bias: str = schema_utils.StringOptions(
        ["none", "all", "lora_only"],
        default="none",
        allow_none=True,
        description="This bias type for Lora. Can be `none`, `all` or `lora_only`",
    )

    modules_to_save: Optional[List[str]] = schema_utils.List(
        list_type=str,
        default=None,
        allow_none=True,
        description="List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        "For example, in Sequence Classification or Token Classification tasks, "
        "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.",
    )

    init_lora_weights: bool = schema_utils.Boolean(
        default=True,
        allow_none=True,
        description="Whether to initialize LoRA weights with the original weights of the layer to replace.",
    )


@DeveloperAPI
@register_adapter("adalora")
@schema_utils.ludwig_dataclass
class AdaLoRAAdapterConfig(LoRAAdapterConfig):
    """Adapted from https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora.py."""

    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("adalora")

    peft_type: str = schema_utils.ProtectedString("ADALORA")

    target_r: int = schema_utils.Integer(
        default=8,
        allow_none=True,
        description="Target Lora Matrix Dimension. The target average rank of incremental matrix.",
    )

    init_r: int = schema_utils.Integer(
        default=12,
        allow_none=True,
        description="Initial Lora Matrix Dimension. The initial rank for each incremental matrix.",
    )

    tinit: int = schema_utils.Integer(
        default=0,
        allow_none=True,
        description="The steps of initial fine-tuning warmup.",
    )

    tfinal: int = schema_utils.Integer(
        default=0,
        allow_none=True,
        description="The steps of final fine-tuning warmup.",
    )

    deltaT: int = schema_utils.Integer(
        default=1,
        allow_none=True,
        description="The time internval between two budget allocations. The step interval of rank allocation.",
    )

    beta1: float = schema_utils.FloatRange(
        default=0.85,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="The hyperparameter of EMA for sensitivity smoothing.",
    )

    beta2: float = schema_utils.FloatRange(
        default=0.85,
        min=0.0,
        max=1.0,
        allow_none=True,
        description=" The hyperparameter of EMA for undertainty quantification.",
    )

    orth_reg_weight: float = schema_utils.FloatRange(
        default=0.5,
        min=0.0,
        max=1.0,
        allow_none=True,
        description="The coefficient of orthogonality regularization.",
    )

    total_step: Optional[int] = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="The total training steps that should be specified before training.",
    )

    rank_pattern: Optional[dict] = schema_utils.Dict(
        default=None,
        allow_none=True,
        description="The allocated rank for each weight matrix by RankAllocator.",
    )


@DeveloperAPI
def get_adapter_conds():
    """Returns a JSON schema of conditionals to validate against adapter types."""
    conds = []
    for adapter in _adapter_registry:
        adapter_cls = _adapter_registry[adapter]
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(adapter_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": adapter},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def AdapterDataclassField(default=None, description=""):
    class AdapterSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(
                registry=_adapter_registry,
                default_value=default,
                description=description,
            )

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return get_adapter_cls(key)

        def _jsonschema_type_mapping(self):
            return {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": list(_adapter_registry.keys()),
                                "default": default,
                                "description": "The type of adapter to use for LLM fine-tuning",
                            },
                        },
                        "title": "adapter_options",
                        "allOf": get_adapter_conds(),
                        "required": ["type"],
                        "description": description,
                    },
                    {
                        "type": "null",
                    },
                ]
            }

    return AdapterSelection().get_default_field()
