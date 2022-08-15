from marshmallow_dataclass import dataclass

import ludwig.schema.utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class BinaryPassthroughEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["passthrough"],
        default="passthrough",
        allow_none=False,
        description="Type of encoder.",
    )
