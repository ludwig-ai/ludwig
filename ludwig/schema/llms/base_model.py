from dataclasses import field

from marshmallow import fields, ValidationError

from ludwig.api_annotations import DeveloperAPI

# TODO:
# from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json, ParameterMetadata

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
    pm = ParameterMetadata(ui_component_type="radio_string_combined", expected_impact=3)

    class BaseModelField(fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            if isinstance(value, str):
                return value
            raise ValidationError(f"Value to serialize is not a string: {value}")

        def _deserialize(self, value, attr, obj, **kwargs):
            # TODO: Could put huggingface validation here, then could also dovetail preset validation:
            print("DESERIALIZE" * 10)
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
                # validate=lambda x: isinstance(x, str),  # TODO: Could put huggingface validation here
                metadata={
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(pm),
                },  # TODO: Does this matter?
            ),
        },
        default=None,  # TODO: Unfortunate side-effect of dataclass init order, super ugly
    )
