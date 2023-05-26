from abc import ABC
from typing import Optional, Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import ParameterMetadata
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
    preset: Optional[str]
    name: str


@DeveloperAPI
@register_base_model(name="none")
@ludwig_dataclass
class NoPresetModelConfig(BaseModelConfig):
    def __post_init__(self):
        if not self.name:
            raise ConfigValidationError(
                "LLM requires `base_model.name` to be set. This can be any pretrained CausalLM on huggingface. "
                "See: https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads"
            )

    preset: Optional[str] = schema_utils.ProtectedString(
        "none", parameter_metadata=LLM_METADATA["base_model"]["default"]["preset"]
    )

    name: str = schema_utils.String(
        default=None,
        allow_none=True,
        description=(
            "The name of the model to use. This can be a local path or a "
            "remote path. If it is a remote path, it must be a valid HuggingFace "
            "model name. If it is a local path, it must be a valid HuggingFace "
            "model name or a path to a local directory containing a valid "
            "HuggingFace model."
        ),
        parameter_metadata=LLM_METADATA["base_model"]["none"]["name"],
    )


for preset, model_name in MODEL_PRESETS.items():

    @DeveloperAPI
    @register_base_model(name=preset)
    @ludwig_dataclass
    class PresetModelConfig(BaseModelConfig):
        preset: Optional[str] = schema_utils.ProtectedString(
            preset, parameter_metadata=LLM_METADATA["base_model"]["default"]["preset"]
        )

        name: str = schema_utils.ProtectedString(
            model_name, parameter_metadata=LLM_METADATA["base_model"]["default"]["name"]
        )


@DeveloperAPI
def get_base_model_conds():
    conds = []
    for base_model_type, base_model_cls in base_model_registry.items():
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(base_model_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props, fields=["preset"])
        preproc_cond = schema_utils.create_cond(
            {"preset": base_model_type},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


@DeveloperAPI
def BaseModelDataclassField(
    default: Optional[str] = None, description: str = "", parameter_metadata: ParameterMetadata = None
):
    class BaseModelSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(
                registry=base_model_registry,
                default_value=default,
                key="preset",
                description=description,
                parameter_metadata=parameter_metadata,
                allow_str_value=True,
                allow_none=True,
            )

        def str_value_to_object(self, value: str) -> str:
            if value in MODEL_PRESETS:
                # User provided a model preset name, so use it
                return {self.key: value}

            # Otherwise, assume the user is providing a fully qualified model name
            return {self.key: "none", "name": value}

        def get_schema_from_registry(self, key: Optional[str]) -> Type[schema_utils.BaseMarshmallowConfig]:
            return base_model_registry[key]

        def _jsonschema_type_mapping(self):
            return {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "preset": {
                                "type": "string",
                                "enum": list(base_model_registry.keys()),
                                "default": default,
                                "description": "MISSING",
                            },
                        },
                        "title": "base_model_object_options",
                        "allOf": get_base_model_conds(),
                        "required": ["preset"],
                        "description": description,
                    },
                    {"type": "string", "title": "base_model_string_options", "description": "MISSING"},
                    {"type": "null", "title": "base_model_null_option", "description": "MISSING"},
                ],
                "title": "base_model_options",
                "description": "The type of base models to use for the LLM",
            }

    return BaseModelSelection().get_default_field()
