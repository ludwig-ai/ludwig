from marshmallow_dataclass import dataclass

import ludwig.schema.utils as schema_utils
from ludwig.constants import BINARY
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


@register_encoder_config("passthrough", BINARY)
@dataclass
class BinaryPassthroughEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["passthrough"],
        default="passthrough",
        allow_none=False,
        description="Type of encoder.",
    )
