import logging

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.image.pretrained import (
    CLIPImageEncoderConfig,
    DINOv2ImageEncoderConfig,
    SigLIPImageEncoderConfig,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("clip", IMAGE)
class CLIPImageEncoder(ImageEncoder):
    """CLIP image encoder (Radford et al., ICML 2021).

    Encodes images using CLIP's vision transformer. Produces embeddings aligned with
    text in a shared latent space, enabling zero-shot classification and multimodal tasks.

    Use when: zero-shot image classification, image-text retrieval, or multimodal
    fusion where visual-semantic alignment matters.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        from transformers import CLIPVisionConfig, CLIPVisionModel

        if use_pretrained and not saved_weights_in_checkpoint:
            logger.info(f"Loading pretrained CLIP vision model: {pretrained_model_name_or_path}")
            self.model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
        else:
            logger.info("Instantiating CLIP vision model without pretrained weights.")
            self.model = CLIPVisionModel(CLIPVisionConfig())

        self._output_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        outputs = self.model(pixel_values=inputs)
        return {ENCODER_OUTPUT: outputs.pooler_output}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return CLIPImageEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        size = self.model.config.image_size
        return torch.Size([3, size, size])


@DeveloperAPI
@register_encoder("dinov2", IMAGE)
class DINOv2ImageEncoder(ImageEncoder):
    """DINOv2 image encoder (Oquab et al., TMLR 2024).

    Self-supervised visual features that work well as frozen backbones for dense prediction
    and linear probing. No text alignment needed.

    Use when: image classification/segmentation without labels for pretraining,
    dense prediction tasks, or as a general-purpose frozen feature extractor.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/dinov2-base",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        from transformers import Dinov2Config, Dinov2Model

        if use_pretrained and not saved_weights_in_checkpoint:
            logger.info(f"Loading pretrained DINOv2 model: {pretrained_model_name_or_path}")
            self.model = Dinov2Model.from_pretrained(pretrained_model_name_or_path)
        else:
            logger.info("Instantiating DINOv2 model without pretrained weights.")
            self.model = Dinov2Model(Dinov2Config())

        self._output_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        outputs = self.model(pixel_values=inputs)
        return {ENCODER_OUTPUT: outputs.pooler_output}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return DINOv2ImageEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        size = self.model.config.image_size
        return torch.Size([3, size, size])


@DeveloperAPI
@register_encoder("siglip", IMAGE)
class SigLIPImageEncoder(ImageEncoder):
    """SigLIP image encoder (Zhai et al., ICCV 2023).

    Uses sigmoid loss instead of softmax for image-text pre-training, enabling
    better scaling and more efficient batch processing than CLIP.

    Use when: similar to CLIP but with better scaling properties,
    or when using SigLIP-specific pretrained models.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/siglip-base-patch16-224",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        from transformers import SiglipVisionConfig, SiglipVisionModel

        if use_pretrained and not saved_weights_in_checkpoint:
            logger.info(f"Loading pretrained SigLIP vision model: {pretrained_model_name_or_path}")
            self.model = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path)
        else:
            logger.info("Instantiating SigLIP vision model without pretrained weights.")
            self.model = SiglipVisionModel(SiglipVisionConfig())

        self._output_dim = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        outputs = self.model(pixel_values=inputs)
        return {ENCODER_OUTPUT: outputs.pooler_output}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return SigLIPImageEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        size = self.model.config.image_size
        return torch.Size([3, size, size])
