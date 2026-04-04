from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


class PretrainedImageEncoderConfig(BaseEncoderConfig):
    """Base config for HuggingFace pretrained image encoders."""

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Whether to use pretrained weights from HuggingFace.",
    )
    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether encoder parameters are trainable.",
    )
    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description=(
            "Are the pretrained encoder weights saved in this model's checkpoint? "
            "Automatically set to True for trained models to prevent loading pretrained "
            "encoder weights from the model hub."
        ),
    )

    def is_pretrained(self) -> bool:
        return self.use_pretrained


@DeveloperAPI
@register_encoder_config("clip", IMAGE)
class CLIPImageEncoderConfig(PretrainedImageEncoderConfig):
    @staticmethod
    def module_name():
        return "CLIPImageEncoder"

    type: str = schema_utils.ProtectedString(
        "clip",
        description="CLIP image encoder using a pretrained vision transformer from OpenAI.",
    )
    pretrained_model_name_or_path: str = schema_utils.String(
        default="openai/clip-vit-base-patch32",
        description="HuggingFace model path or name for the CLIP vision model.",
    )


@DeveloperAPI
@register_encoder_config("dinov2", IMAGE)
class DINOv2ImageEncoderConfig(PretrainedImageEncoderConfig):
    @staticmethod
    def module_name():
        return "DINOv2ImageEncoder"

    type: str = schema_utils.ProtectedString(
        "dinov2",
        description="DINOv2 image encoder using self-supervised visual features from Meta.",
    )
    pretrained_model_name_or_path: str = schema_utils.String(
        default="facebook/dinov2-base",
        description="HuggingFace model path or name for the DINOv2 model.",
    )


@DeveloperAPI
@register_encoder_config("siglip", IMAGE)
class SigLIPImageEncoderConfig(PretrainedImageEncoderConfig):
    @staticmethod
    def module_name():
        return "SigLIPImageEncoder"

    type: str = schema_utils.ProtectedString(
        "siglip",
        description="SigLIP image encoder using sigmoid loss for image-text pre-training from Google.",
    )
    pretrained_model_name_or_path: str = schema_utils.String(
        default="google/siglip-base-patch16-224",
        description="HuggingFace model path or name for the SigLIP vision model.",
    )
