from marshmallow_dataclass import dataclass

import ludwig.schema.utils as schema_utils
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, MODEL_ECD, MODEL_GBM
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


@DeveloperAPI
@register_encoder_config("passthrough", BINARY, model_types=[MODEL_ECD, MODEL_GBM])
@dataclass(repr=False)
class BinaryPassthroughEncoderConfig(BaseEncoderConfig):
    type: str = schema_utils.ProtectedString(
        "passthrough",
        description="Type of encoder.",
    )
