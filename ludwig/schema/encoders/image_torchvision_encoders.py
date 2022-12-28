from typing import Optional, Union

from marshmallow_dataclass import dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA


@dataclass
class TVBaseEncoderConfig(BaseEncoderConfig):
    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Download model weights from pre-trained model.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["use_pretrained"],
    )

    model_cache_dir: Optional[str] = schema_utils.String(
        description="Directory path to cache pretrained model weights.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["model_cache_dir"],
    )

    saved_weights_in_checkpoint: bool = schema_utils.Boolean(
        default=False,
        description="Whether to save the weights in the checkpoint.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["saved_weights_in_checkpoint"],
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="Is the encoder trainable.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["trainable"],
    )


@DeveloperAPI
@register_encoder_config("alexnet_torch", IMAGE)
@dataclass
class TVAlexNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("alexnet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVAlexNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("convnext_torch", IMAGE)
@dataclass
class TVConvNeXtEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("convnext_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["tiny", "small", "base", "large"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVConvNeXtEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("densenet_torch", IMAGE)
@dataclass
class TVDenseNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("densenet_torch", description="Type of encoder.")

    model_variant: int = schema_utils.IntegerOptions(
        [121, 161, 169, 201],
        default=121,
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVDenseNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("efficientnet_torch", IMAGE)
@dataclass
class TVEfficientNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("efficientnet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "b0",
            "b1",
            "b2",
            "b3",
            "b4",
            "b5",
            "b6",
            "b7",
            "v2_",
            "v2_m",
            "v2_l",
        ],
        default="b0",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVEfficientNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("googlenet_torch", IMAGE)
@dataclass
class TVGoogLeNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("googlenet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVGoogLeNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("inceptionv3_torch", IMAGE)
@dataclass
class TVInceptionV3EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("inceptionv3_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVGoogLeNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("maxvit_torch", IMAGE)
@dataclass
class TVMaxVitEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("maxvit_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["t"],
        default="t",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMNASNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("mnasnet_torch", IMAGE)
@dataclass
class TVMNASNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("mnasnet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["0_5", "0_75", "1_0", "1_3"],
        default="0_5",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMNASNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("mobilenetv2_torch", IMAGE)
@dataclass
class TVMobileNetV2EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("mobilenetv2_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMobileNetV2Encoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("mobilenetv3_torch", IMAGE)
@dataclass
class TVMobileNetV3EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("mobilenetv3_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "small",
            "large",
        ],
        default="small",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMobileNetV3Encoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("regnet_torch", IMAGE)
@dataclass
class TVRegNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("regnet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "x_1_6gf",
            "x_16gf",
            "x_32gf",
            "x_3_2gf",
            "x_400mf",
            "x_800mf",
            "x_8gf",
            "y_128gf",
            "y_16gf",
            "y_1_6gf",
            "y_32gf",
            "y_3_2gf",
            "y_400mf",
            "y_800mf",
            "y_8gf",
        ],
        default="x_1_6gf",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVRegNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("resnet_torch", IMAGE)
@dataclass
class TVResNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("resnet_torch", description="Type of encoder.")

    model_variant: int = schema_utils.IntegerOptions(
        [18, 34, 50, 101, 152],
        default=50,
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVResNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("resnext_torch", IMAGE)
@dataclass
class TVResNeXtEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("resnext_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["50_32x4d", "101_32x8d", "101_64x4d"],
        default="50_32x4d",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVResNeXtEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("shufflenet_v2_torch", IMAGE)
@dataclass
class TVShuffleNetV2EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("shufflenet_v2_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "x0_5",
            "x1_0",
            "x1_5",
            "x2_0",
        ],
        default="x0_5",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVShuffleNetV2Encoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("squeezenet_torch", IMAGE)
@dataclass
class TVSqueezeNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("squeezenet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "1_0",
            "1_1",
        ],
        default="1_0",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVSqueezeNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("swin_transformer_torch", IMAGE)
@dataclass
class TVSwinTransformerEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("swin_transformer_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "t",
            "s",
            "b",
        ],
        default="t",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVSwinTransformerEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("vit_torch", IMAGE)
@dataclass
class TVViTEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("vit_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "b_16",
            "b_32",
            "l_16",
            "l_32",
            "h_14",
        ],
        default="b_16",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVViTEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("vgg_torch", IMAGE)
@dataclass
class TVVGGEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("vgg_torch", description="Type of encoder.")

    model_variant: Union[int, str] = schema_utils.OneOfOptionsField(
        default=11,
        description="Pretrained model variant to use.",
        field_options=[
            schema_utils.IntegerOptions(
                [
                    11,
                    13,
                    16,
                    19,
                ],
                default=11,
                allow_none=False,
            ),
            schema_utils.StringOptions(
                [
                    "11_bn",
                    "13_bn",
                    "16_bn",
                    "19_bn",
                ],
                default="11_bn",
                allow_none=False,
            ),
        ],
        allow_none=False,
        parameter_metadata=ENCODER_METADATA["TVVGGEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("wide_resnet_torch", IMAGE)
@dataclass
class TVWideResNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("wide_resnet_torch", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        [
            "50_2",
            "101_2",
        ],
        default="50_2",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVViTEncoder"]["model_variant"],
    )
