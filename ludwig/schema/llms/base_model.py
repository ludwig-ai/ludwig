from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BASE_MODEL
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json

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


@DeveloperAPI
def BaseModelDataclassField(
    description: str = "",
):
    class BaseModelField(fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            if isinstance(value, str):
                return value
            raise ValidationError(f"Value to serialize is not a string: {value}")

        def _deserialize(self, value, attr, obj, **kwargs):
            # TODO: Could put huggingface validation here, then could also dovetail preset validation:
            if isinstance(value, str):
                return value
            raise ValidationError(f"Value to deserialize is not a string: {value}")

        def _jsonschema_type_mapping(self):
            return {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": list(MODEL_PRESETS.keys()),
                        "description": "Pick an LLM with first-class Ludwig support.",
                        "title": "preset",
                        "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]),
                    },
                    {
                        "type": "string",
                        "description": "Enter the full (slash-delimited) path to a huggingface LLM.",
                        "title": "custom",
                        "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]),
                    },
                ],
                "description": description,
                "title": "base_model_options",
                "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]),
            }

    return field(
        metadata={
            "marshmallow_field": BaseModelField(
                required=True,
                allow_none=False,
                # validate=lambda x: isinstance(x, str),  # TODO: Could put huggingface validation here
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]),
                },  # TODO: extra metadata dict probably unnecessary, but keep it because it's a widespread pattern.
            ),
        },
        default=None,  # TODO: Unfortunate side-effect of dataclass init order, super ugly
    )
