import logging
from typing import Optional, Type

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TEXT
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.base import BaseEncoderConfig, PassthroughEncoderConfig

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("external", [TEXT])
class ExternalEncoder(Encoder):
    def __init__(self, input_size=1, encoder_config=None, **kwargs):
        super().__init__()
        self.config = encoder_config
        self.input_size = input_size
        self.type = "external"
        self.skip = False

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderOutputDict:
        raise NotImplementedError()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return PassthroughEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape
