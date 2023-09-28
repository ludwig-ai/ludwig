import os
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
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-70b": "meta-llama/Llama-2-70b-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "vicuna-7b": "lmsys/vicuna-7b-v1.3",
    "vicuna-13b": "lmsys/vicuna-13b-v1.3",
    "bloomz-3b": "bigscience/bloomz-3b",
    "bloomz-7b1": "bigscience/bloomz-7b1",
    "opt-350m": "facebook/opt-350m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-6.7b": "facebook/opt-6.7b",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


@DeveloperAPI
def BaseModelDataclassField():
    description = (
        "Base pretrained model to use. This can be one of the presets defined by Ludwig, a fully qualified "
        "name of a pretrained model from the HuggingFace Hub, or a path to a directory containing a "
        "pretrained model."
    )

    def validate(model_name: str):
        """Validates and upgrades the given model name to its full path, if applicable.

        If the name exists in `MODEL_PRESETS`, returns the corresponding value from the dict; otherwise checks if the
        given name (which should be a full path) exists locally or in the transformers library.
        """
        if isinstance(model_name, str):
            if model_name in MODEL_PRESETS:
                return MODEL_PRESETS[model_name]
            if os.path.isdir(model_name):
                return model_name
            try:
                AutoConfig.from_pretrained(model_name)
                return model_name
            except OSError:
                raise ConfigValidationError(
                    f"Specified base model `{model_name}` could not be loaded. If this is a private repository, make "
                    f"sure to set HUGGING_FACE_HUB_TOKEN in your environment. Check that {model_name} is a valid "
                    "pretrained CausalLM listed on huggingface or a valid local directory containing the weights for a "
                    "pretrained CausalLM from huggingface. See: "
                    "https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for a full list."
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
                        "description": (
                            "Pick from a set of popular LLMs of different sizes across a variety of architecture types."
                        ),
                        "title": "preset",
                        "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]["_anyOf"]["preset"]),
                    },
                    {
                        "type": "string",
                        "description": "Enter the full path to a huggingface LLM.",
                        "title": "custom",
                        "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]["_anyOf"]["custom"]),
                    },
                ],
                "description": description,
                "title": "base_model_options",
                "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]["_meta"]),
            }

    return field(
        metadata={
            "marshmallow_field": BaseModelField(
                required=True,
                allow_none=False,
                validate=validate,
                metadata={  # TODO: extra metadata dict probably unnecessary, but currently a widespread pattern
                    "description": description,
                    "parameter_metadata": convert_metadata_to_json(LLM_METADATA[BASE_MODEL]["_meta"]),
                },
            ),
        },
        # TODO: This is an unfortunate side-effect of dataclass init order - you cannot have non-default fields follow
        # default fields, so we have to give `base_model` a fake default of `None`.
        default=None,
    )
