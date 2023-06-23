from abc import ABC
from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils

# from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

# from typing import Optional, Type


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
    description: str = "",
):
    pm = ParameterMetadata(ui_component_type="radio_string_combined", expected_impact=3)

    class BaseModelField(fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            if isinstance(value, str):
                return value
            raise ValidationError(f"Value to serialize is not a string: {value}")

        def _deserialize(self, value, attr, obj, **kwargs):
            # TODO: Could put huggingface validation here, then could also dovetail preset validation:
            if isinstance(value, str):
                return str
            raise ValidationError(f"Value to deserialize is not a string: {value}")

        def _jsonschema_type_mapping(self):
            return {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": list(MODEL_PRESETS.keys()),
                        "description": "Pick an LLM with first-class Ludwig support.",
                        "title": "preset",
                    },
                    {
                        "type": "string",
                        "description": "Enter the full (slash-delimited) path to a huggingface LLM.",
                        "title": "custom",
                    },
                ],
                "description": description,
                "title": "base_model_options",
                "parameter_metadata": convert_metadata_to_json(pm),
            }

    return field(
        metadata={
            "marshmallow_field": BaseModelField(
                required=True,
                allow_none=False,
                validate=lambda x: isinstance(x, str),  # TODO: Could put huggingface validation here
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(pm),
                    # "required": True,
                },
            ),
            # "required": True,
        },
        default=None,  # TODO: Unfortunate side-effect of dataclass init order, super ugly
    )
