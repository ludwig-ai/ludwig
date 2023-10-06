import logging

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.schema.decoders.base import PassthroughDecoderConfig

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_decoder("external", [TEXT])
class ExternalDecoder(Decoder):
    def __init__(self, input_size: int = 1, num_classes: int = None, decoder_config=None, **kwargs):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")
        self.input_size = input_size
        self.num_classes = num_classes

        self.fc_layers = None
        self.fc_output_size = 256
        self.fc_use_bias = (False,)
        self.num_fc_layers = 0
        self.fc_weights_initializer = "xavier_uniform"
        self.fc_bias_initializer = "zeros"
        self.fc_norm = None
        self.fc_norm_params = None
        self.fc_activation = "relu"
        self.fc_dropout = 0
        self.type = "external"

    def forward(self, inputs, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get_schema_cls():
        return PassthroughDecoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape
