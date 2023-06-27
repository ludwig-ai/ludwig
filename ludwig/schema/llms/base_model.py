from dataclasses import field

from marshmallow import fields, ValidationError
from transformers import AutoConfig

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BASE_MODEL
from ludwig.error import ConfigValidationError
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json

# Maps a preset LLM name to the full slash-delimited HF path. If the user chooses a preset LLM, the preset LLM name is
# replaced with the full slash-delimited HF path using this map, after JSON validation but before config object
# initialization.
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
    def validate(model_name: str):
        """Validates and upgrades the given model name to its full path, if applicable.

        If the name exists in `MODEL_PRESETS`, returns the corresponding value from the dict; otherwise checks if the
        given name (which should be a full path) exists in the transformers library.
        """
        if isinstance(model_name, str):
            if model_name in MODEL_PRESETS:
                return MODEL_PRESETS[model_name]
            try:
                AutoConfig.from_pretrained(model_name)
                return model_name
            except OSError:
                raise ConfigValidationError(
                    f"Specified base model `{model_name}` is not a valid pretrained CausalLM listed on huggingface. "
                    "Please see: https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads"
                )
        raise ValidationError(
            f"`base_model` should be a string, instead given: {model_name}. This can be a preset or any pretrained "
            "CausalLM on huggingface. See: https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads"
        )

    class BaseModelField(fields.Field):
        def _serialize(self, value, attr, obj, **kwargs):
            if isinstance(value, str):
                return value
            raise ValidationError(f"Value to serialize is not a string: {value}")

        def _deserialize(self, value, attr, obj, **kwargs):
            return validate(value)

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
                        "description": "Enter the full path to a huggingface LLM.",
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
                validate=validate,
                metadata={  # TODO: extra metadata dict probably unnecessary, but currently a widespread pattern
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]),
                },
            ),
        },
        # TODO: This is an unfortunate side-effect of dataclass init order - you cannot have non-default fields follow
        # default fields, so we have to give `base_model` a fake default of `None`.
        default=None,
    )
