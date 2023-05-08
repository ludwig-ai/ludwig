from abc import ABC
from typing import Optional, Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

_adapter_registry = Registry()


@DeveloperAPI
def register_adapter(name: str):
    """Registers an adapter config class with the adapter config registry."""

    def wrap(adapter_config: BasePeftAdapterConfig):
        _adapter_registry[name] = adapter_config
        return adapter_config

    return wrap


@DeveloperAPI
def get_adapter_cls(name: str):
    """Returns the adapter config class registered with the given name."""
    return _adapter_registry[name]


@DeveloperAPI
@schema_utils.ludwig_dataclass
class BasePeftAdapterConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Config for prompt learning adapters.

    Not meant to be used directly.
    """

    task_type: str = schema_utils.ProtectedString("CAUSAL_LM")

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
class PromptTuningAdapterConfig(BasePeftAdapterConfig):
    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("prompt_tuning")

    peft_type: str = schema_utils.ProtectedString("PROMPT_TUNING")

    # TODO(Arnav): Refactor to allow both RANDOM and TEXT strategies
    prompt_tuning_init: str = schema_utils.ProtectedString(
        "TEXT", description="The type of initialization to use for the prompt embedding. "
    )

    prompt_tuning_init_text: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The text to use to initialize the prompt embedding.",
    )


@DeveloperAPI
@register_adapter("prefix_tuning")
@schema_utils.ludwig_dataclass
class PrefixTuningAdapterconfig(BasePeftAdapterConfig):
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
class PTuningAdapterConfig(BasePeftAdapterConfig):
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
class LoRAAdapterConfig(BasePeftAdapterConfig):
    # Explicitly set type property in the config because it is needed when we
    # load a saved PEFT model back into Ludwig.
    type: str = schema_utils.ProtectedString("lora")

    peft_type: str = schema_utils.ProtectedString("LORA")

    r: int = schema_utils.Integer(
        default=8,
        allow_none=True,
        description="LoRA attention dimension",
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
