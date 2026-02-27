from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@ludwig_dataclass
class TimmBaseConfig(BaseEncoderConfig):
    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Download model weights from pretrained model.",
        parameter_metadata=ENCODER_METADATA["TimmEncoder"]["use_pretrained"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to use weights saved in the Ludwig checkpoint instead of pretrained weights.",
        parameter_metadata=ENCODER_METADATA["TimmEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Whether the encoder parameters are trainable.",
        parameter_metadata=ENCODER_METADATA["TimmEncoder"]["trainable"],
    )

    def is_pretrained(self) -> bool:
        return self.use_pretrained


@DeveloperAPI
@register_encoder_config("timm", IMAGE)
@ludwig_dataclass
class TimmEncoderConfig(TimmBaseConfig):
    type: str = schema_utils.ProtectedString("timm", description="Type of encoder.")

    model_name: str = schema_utils.String(
        default="caformer_s18",
        description=(
            "Name of the timm model to use. Any model from the timm library is supported. "
            "See https://huggingface.co/docs/timm for available models."
        ),
        parameter_metadata=ENCODER_METADATA["TimmEncoder"]["model_name"],
    )


# Convenience aliases for MetaFormer variants with curated model_name options

CAFORMER_MODELS = [
    "caformer_s18",
    "caformer_s36",
    "caformer_m36",
    "caformer_b36",
    "caformer_s18.sail_in22k_ft_in1k",
    "caformer_s18.sail_in22k_ft_in1k_384",
    "caformer_s36.sail_in22k_ft_in1k",
    "caformer_s36.sail_in22k_ft_in1k_384",
    "caformer_m36.sail_in22k_ft_in1k",
    "caformer_m36.sail_in22k_ft_in1k_384",
    "caformer_b36.sail_in22k_ft_in1k",
    "caformer_b36.sail_in22k_ft_in1k_384",
]

CONVFORMER_MODELS = [
    "convformer_s18",
    "convformer_s36",
    "convformer_m36",
    "convformer_b36",
    "convformer_s18.sail_in22k_ft_in1k",
    "convformer_s18.sail_in22k_ft_in1k_384",
    "convformer_s36.sail_in22k_ft_in1k",
    "convformer_s36.sail_in22k_ft_in1k_384",
    "convformer_m36.sail_in22k_ft_in1k",
    "convformer_m36.sail_in22k_ft_in1k_384",
    "convformer_b36.sail_in22k_ft_in1k",
    "convformer_b36.sail_in22k_ft_in1k_384",
]

POOLFORMER_MODELS = [
    "poolformerv2_s12",
    "poolformerv2_s24",
    "poolformerv2_s36",
    "poolformerv2_m36",
    "poolformerv2_m48",
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformer_m36",
    "poolformer_m48",
]


@DeveloperAPI
@register_encoder_config("caformer", IMAGE)
@ludwig_dataclass
class TimmCAFormerEncoderConfig(TimmBaseConfig):
    type: str = schema_utils.ProtectedString("caformer", description="Type of encoder.")

    model_name: str = schema_utils.StringOptions(
        CAFORMER_MODELS,
        default="caformer_s18",
        allow_none=False,
        description=(
            "CAFormer model variant. Hybrid Conv+Attention MetaFormer achieving SOTA accuracy. "
            "Variants with '.sail_in22k_ft_in1k' are pretrained on ImageNet-21K and finetuned on ImageNet-1K. "
            "Variants with '_384' use 384x384 input resolution."
        ),
        parameter_metadata=ENCODER_METADATA["TimmCAFormerEncoder"]["model_name"],
    )


@DeveloperAPI
@register_encoder_config("convformer", IMAGE)
@ludwig_dataclass
class TimmConvFormerEncoderConfig(TimmBaseConfig):
    type: str = schema_utils.ProtectedString("convformer", description="Type of encoder.")

    model_name: str = schema_utils.StringOptions(
        CONVFORMER_MODELS,
        default="convformer_s18",
        allow_none=False,
        description=(
            "ConvFormer model variant. Pure CNN MetaFormer that outperforms ConvNeXt. "
            "Variants with '.sail_in22k_ft_in1k' are pretrained on ImageNet-21K and finetuned on ImageNet-1K."
        ),
        parameter_metadata=ENCODER_METADATA["TimmConvFormerEncoder"]["model_name"],
    )


@DeveloperAPI
@register_encoder_config("poolformer", IMAGE)
@ludwig_dataclass
class TimmPoolFormerEncoderConfig(TimmBaseConfig):
    type: str = schema_utils.ProtectedString("poolformer", description="Type of encoder.")

    model_name: str = schema_utils.StringOptions(
        POOLFORMER_MODELS,
        default="poolformerv2_s12",
        allow_none=False,
        description=(
            "PoolFormer model variant. MetaFormer using simple average pooling as token mixer. "
            "V2 variants use StarReLU activation and improved training recipe."
        ),
        parameter_metadata=ENCODER_METADATA["TimmPoolFormerEncoder"]["model_name"],
    )
