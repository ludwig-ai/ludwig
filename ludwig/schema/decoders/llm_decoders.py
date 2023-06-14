from typing import Any, Dict

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, MODEL_LLM, TEXT
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.utils import BaseMarshmallowConfig, ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class BaseExtractorDecoderConfig(BaseMarshmallowConfig):
    tokenizer: str = "hf_tokenizer"

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
    )

    pretrained_model_name_or_path: str = schema_utils.String(
        default="",
        allow_none=True,
        description="Path to the pretrained model or model identifier from huggingface.co/models.",
    )

    vocab_file: str = schema_utils.String(
        default="",
        allow_none=True,
        description="Path to the vocabulary file.",
    )

    max_new_tokens: int = schema_utils.Integer(
        default=None,
        allow_none=True,
        description="Maximum number of new tokens that will be generated.",
    )


@DeveloperAPI
@register_decoder_config("text_extractor", [TEXT], model_types=[MODEL_LLM])
@ludwig_dataclass
class TextExtractorDecoderConfig(BaseExtractorDecoderConfig, BaseDecoderConfig):
    @classmethod
    def module_name(cls):
        return "TextExtractorDecoder"

    type: str = schema_utils.ProtectedString("text_extractor")


@DeveloperAPI
@register_decoder_config("category_extractor", [CATEGORY], model_types=[MODEL_LLM])
@ludwig_dataclass
class CategoryExtractorDecoderConfig(BaseExtractorDecoderConfig, BaseDecoderConfig):
    @classmethod
    def module_name(cls):
        return "CategoryExtractorDecoder"

    type: str = schema_utils.ProtectedString("category_extractor")

    # Match is a dict of label class
    match: Dict[str, Dict[str, Any]] = schema_utils.Dict(
        default=None,
        allow_none=False,
        description="A dictionary of label classes and their corresponding "
        "match patterns definitions that will be used to parse the output "
        "of the LLM.",
    )

    str2idx: Dict[str, int] = schema_utils.Dict(
        default=None,
        allow_none=True,
        description="A dictionary of label classes and their corresponding "
        "indices that will be used to parse the output of the LLM.",
    )

    fallback_label: str = schema_utils.String(
        default="",
        allow_none=True,
        description="The label to use if the parser fails to parse the input.",
    )
