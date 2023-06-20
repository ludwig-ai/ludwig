from abc import ABC
from typing import Optional, Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils

# from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

MODEL_PRESETS = {
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "bloomz-3b": "bigscience/bloomz-3b",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "stablelm-base-alpha-3b": "stabilityai/stablelm-base-alpha-3b",
    "llama-7b": "huggyllama/llama-7b",
    "vicuna-7b": "eachadea/vicuna-7b-1.1",
    "bloom-7b": "bigscience/bloom-7b1",
    "stablelm-base-alpha-7b": "stabilityai/stablelm-base-alpha-7b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "oasst-sft-1-pythia-12b": "OpenAssistant/oasst-sft-1-pythia-12b",
    "vicuna-13b": "eachadea/vicuna-13b-1.1",
}


base_model_registry = Registry()


@DeveloperAPI
def register_base_model(name: str):
    def wrap(config: BaseModelConfig):
        base_model_registry[name] = config
        return config

    return wrap


@DeveloperAPI
@ludwig_dataclass
class BaseModelConfig(schema_utils.BaseMarshmallowConfig, ABC):
    type: str = schema_utils.StringOptions(
        ["preset", "custom"], default="preset", description="TODO", parameter_metadata=None
    )  # TODO: should it include none?


@DeveloperAPI
@base_model_registry.register("preset")
@ludwig_dataclass
class BaseModelPresetConfig(BaseModelConfig):
    type: str = schema_utils.ProtectedString("preset")

    name: str = schema_utils.StringOptions(MODEL_PRESETS.keys(), default="vicuna-13b")


@DeveloperAPI
@base_model_registry.register("custom")
@ludwig_dataclass
class BaseModelCustomConfig(BaseModelConfig):
    type: str = schema_utils.ProtectedString("custom")

    name: str = schema_utils.String(description="TODO", default="TODO")

    def __post_init__(self):
        if not self.name:
            raise ConfigValidationError(
                "Customized LLM requires `base_model.name` to set. This can be any pretrained CausalLM on huggingface. "
                "See: https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads"
            )


@DeveloperAPI
def get_base_model_conds():
    conds = []
    for base_model_type, base_model_cls in base_model_registry.items():
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(base_model_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props, fields=["type"])  # do not remove 'name'
        # TODO: probably can improve the deduplication logic
        preproc_cond = schema_utils.create_cond(
            {"type": base_model_type},
            other_props,
        )
        conds.append(preproc_cond)

    # Make `name` required for the custom case:
    conds[1]["then"]["required"] = ["name"]
    conds[1]["then"]["properties"].pop("default", None)

    return conds


@DeveloperAPI
def BaseModelDataclassField(
    default: Optional[str] = None,
    description: str = "",
):
    pm = ParameterMetadata(ui_component_type="radio_string_combined")

    class BaseModelSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(
                registry=base_model_registry,
                default_value=default,
                key="type",
                description=description,
                parameter_metadata=pm,
                # allow_str_value=True,
                allow_none=True,
            )

        def get_schema_from_registry(self, key: Optional[str]) -> Type[schema_utils.BaseMarshmallowConfig]:
            return base_model_registry[key]

        def _jsonschema_type_mapping(self):
            return (
                {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": list(base_model_registry.keys()),
                            "default": default,
                            "description": "TODO",
                        },
                    },
                    "title": "base_model_options",
                    "allOf": get_base_model_conds(),
                    "required": ["type"],
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(pm),
                },
            )

    return BaseModelSelection().get_default_field()
