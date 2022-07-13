from marshmallow_dataclass import dataclass
from ludwig.schema import utils as schema_utils


@dataclass
class BinaryPassthroughEncoderConfig(schema_utils.BaseMarshmallowConfig):

    type: str = "passthrough"
