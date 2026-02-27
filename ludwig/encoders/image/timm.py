import logging

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.image.timm import (
    TimmCAFormerEncoderConfig,
    TimmConvFormerEncoderConfig,
    TimmEncoderConfig,
    TimmPoolFormerEncoderConfig,
)

logger = logging.getLogger(__name__)


def _get_timm():
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for this encoder. Install it with: pip install timm")
    return timm


@DeveloperAPI
@register_encoder("timm", IMAGE)
class TimmEncoder(ImageEncoder):
    """Wraps any model from the timm (pytorch-image-models) library as a Ludwig image encoder.

    This provides access to hundreds of pretrained vision models including MetaFormer variants
    (CAFormer, ConvFormer, PoolFormer), ConvNeXt V2, EfficientFormer, and many more.

    Usage in Ludwig config:
        encoder:
            type: timm
            model_name: caformer_s18.sail_in22k_ft_in1k
            use_pretrained: true
            trainable: true
    """

    def __init__(
        self,
        model_name: str = "caformer_s18",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        logger.debug(f" {self.name}")

        timm = _get_timm()

        pretrained = use_pretrained and not saved_weights_in_checkpoint
        if pretrained:
            logger.info(f"Instantiating timm image encoder '{model_name}' with pretrained weights.")
        else:
            logger.info(f"Instantiating timm image encoder '{model_name}' without pretrained weights.")

        # num_classes=0 removes the classification head, returning pooled features
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Get the model's expected input config for input_shape
        data_config = timm.data.resolve_model_data_config(self.model)
        self._input_size = data_config["input_size"]  # (C, H, W)

        # Compute output dim by running a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, *self._input_size)
            out = self.model(dummy)
            self._output_dim = out.shape[-1]

        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        return {ENCODER_OUTPUT: self.model(inputs)}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return TimmEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_size)


@DeveloperAPI
@register_encoder("caformer", IMAGE)
class TimmCAFormerEncoder(TimmEncoder):
    """CAFormer encoder — hybrid Conv+Attention MetaFormer achieving SOTA accuracy on ImageNet.

    Variants: s18 (26M, 83.6%), s36 (39M, 84.5%), m36 (56M, 85.2%), b36 (99M, 85.5%).
    """

    def __init__(self, model_name: str = "caformer_s18", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return TimmCAFormerEncoderConfig


@DeveloperAPI
@register_encoder("convformer", IMAGE)
class TimmConvFormerEncoder(TimmEncoder):
    """ConvFormer encoder — pure CNN MetaFormer that outperforms ConvNeXt."""

    def __init__(self, model_name: str = "convformer_s18", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return TimmConvFormerEncoderConfig


@DeveloperAPI
@register_encoder("poolformer", IMAGE)
class TimmPoolFormerEncoder(TimmEncoder):
    """PoolFormer encoder — MetaFormer using simple average pooling as token mixer."""

    def __init__(self, model_name: str = "poolformerv2_s12", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return TimmPoolFormerEncoderConfig
