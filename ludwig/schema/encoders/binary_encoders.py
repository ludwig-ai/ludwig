from marshmallow_dataclass import dataclass

from ludwig.schema.encoders.base import BaseEncoderConfig
import ludwig.schema.utils as schema_utils


@dataclass
class BinaryPassthroughEncoderConfig(BaseEncoderConfig):

    type: str = schema_utils.StringOptions(
        ["passthrough"],
        default="passthrough",
        allow_none=False,
        description="Type of encoder.",
    )
