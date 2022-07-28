from marshmallow_dataclass import dataclass

from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class BinaryPassthroughEncoderConfig(BaseEncoderConfig):

    type: str = "passthrough"
