from abc import ABC, abstractmethod
from typing import Optional, Type, TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

if TYPE_CHECKING:
    from peft import PeftConfig


tuner_registry = Registry()


@DeveloperAPI
def register_tuner(name: str):
    def wrap(config: BaseTunerConfig):
        tuner_registry[name] = config
        return config

    return wrap


@DeveloperAPI
@ludwig_dataclass
class BaseTunerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    type: str

    @abstractmethod
    def to_config(self) -> "PeftConfig":
        pass


@DeveloperAPI
@register_tuner(name="lora")
@ludwig_dataclass
class LoraConfig(BaseTunerConfig):
    type: str = schema_utils.ProtectedString("lora")

    r: int = schema_utils.PositiveInteger(
        default=8,
        description="Lora attention dimension.",
    )

    alpha: int = schema_utils.PositiveInteger(
        default=16,
        description="The alpha parameter for Lora scaling.",
    )

    dropout: float = schema_utils.NonNegativeFloat(
        default=0.05,
        description="The dropout probability for Lora layers.",
    )

    # TODO(travis): figure out why calling this `bias` doesn't work
    bias_type: str = schema_utils.StringOptions(
        options=["none", "all", "lora_only"],
        default="none",
        description="Bias type for Lora.",
    )

    def to_config(self) -> "PeftConfig":
        from peft import LoraConfig as _LoraConfig

        return _LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias_type,
        )


@DeveloperAPI
def get_tuner_conds():
    conds = []
    for tuner_type, tuner_cls in tuner_registry.items():
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(tuner_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": tuner_type},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def TunerDataclassField(
    default: Optional[str] = None, description: str = "", parameter_metadata: ParameterMetadata = None
):
    class TunerSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(
                registry=tuner_registry,
                default_value=default,
                description=description,
                parameter_metadata=parameter_metadata,
                allow_str_value=True,
                allow_none=True,
            )

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return tuner_registry[key]

        def _jsonschema_type_mapping(self):
            return {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": list(tuner_registry.keys()),
                                "default": default,
                                "description": "MISSING",
                            },
                        },
                        "title": "tuner_object_options",
                        "allOf": get_tuner_conds(),
                        "required": ["type"],
                        "description": description,
                    },
                    {"type": "string", "title": "tuner_string_options", "description": "MISSING"},
                    {"type": "null", "title": "tuner_null_option", "description": "MISSING"},
                ],
                "title": "tuner_options",
                "description": "The type of PEFT tuner to use during fine-tuning",
            }

    return TunerSelection().get_default_field()
