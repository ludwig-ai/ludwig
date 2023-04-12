from typing import Any, Dict

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_decoder_config("parser", [CATEGORY])
@ludwig_dataclass
class ParserDecoderConfig(BaseDecoderConfig):
    @classmethod
    def module_name(cls):
        return "ParserDecoder"

    type: str = schema_utils.ProtectedString("parser")

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
    )

    # Match is a dict of label class
    match: Dict[str, Dict[str, Any]] = schema_utils.Dict(
        default=None,
        allow_none=False,
    )

    tokenizer: str = "hf_tokenizer"

    pretrained_model_name_or_path: str = ""

    vocab_file: str = None

    str2idx: Dict[str, int] = schema_utils.Dict(
        default=None,
        allow_none=True,
    )

    fallback_label: str = None
