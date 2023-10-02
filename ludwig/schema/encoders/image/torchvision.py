from typing import Optional, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@ludwig_dataclass
class TVBaseEncoderConfig(BaseEncoderConfig):
    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="Download model weights from pre-trained model.",
        parameter_metadata=ENCODER_METADATA["TVBaseEncoder"]["use_pretrained"],
    )

    model_cache_dir: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
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

    def is_pretrained(self) -> bool:
        return self.use_pretrained


@DeveloperAPI
@register_encoder_config("alexnet", IMAGE)
@ludwig_dataclass
class TVAlexNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("alexnet", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVAlexNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("convnext", IMAGE)
@ludwig_dataclass
class TVConvNeXtEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("convnext", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["tiny", "small", "base", "large"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVConvNeXtEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("densenet", IMAGE)
@ludwig_dataclass
class TVDenseNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("densenet", description="Type of encoder.")

    model_variant: int = schema_utils.IntegerOptions(
        [121, 161, 169, 201],
        default=121,
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVDenseNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("efficientnet", IMAGE)
@ludwig_dataclass
class TVEfficientNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("efficientnet", description="Type of encoder.")

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
            "v2_s",
            "v2_m",
            "v2_l",
        ],
        default="b0",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVEfficientNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("googlenet", IMAGE)
@ludwig_dataclass
class TVGoogLeNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("googlenet", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVGoogLeNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("inceptionv3", IMAGE)
@ludwig_dataclass
class TVInceptionV3EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("inceptionv3", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVGoogLeNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("maxvit", IMAGE)
@ludwig_dataclass
class TVMaxVitEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("maxvit", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["t"],
        default="t",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMNASNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("mnasnet", IMAGE)
@ludwig_dataclass
class TVMNASNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("mnasnet", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["0_5", "0_75", "1_0", "1_3"],
        default="0_5",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMNASNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("mobilenetv2", IMAGE)
@ludwig_dataclass
class TVMobileNetV2EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("mobilenetv2", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["base"],
        default="base",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVMobileNetV2Encoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("mobilenetv3", IMAGE)
@ludwig_dataclass
class TVMobileNetV3EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("mobilenetv3", description="Type of encoder.")

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
@register_encoder_config("regnet", IMAGE)
@ludwig_dataclass
class TVRegNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("regnet", description="Type of encoder.")

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
@register_encoder_config("resnet", IMAGE)
@ludwig_dataclass
class TVResNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("resnet", description="Type of encoder.")

    model_variant: int = schema_utils.IntegerOptions(
        [18, 34, 50, 101, 152],
        default=50,
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVResNetEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("resnext", IMAGE)
@ludwig_dataclass
class TVResNeXtEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("resnext", description="Type of encoder.")

    model_variant: str = schema_utils.StringOptions(
        ["50_32x4d", "101_32x8d", "101_64x4d"],
        default="50_32x4d",
        allow_none=False,
        description="Pretrained model variant to use.",
        parameter_metadata=ENCODER_METADATA["TVResNeXtEncoder"]["model_variant"],
    )


@DeveloperAPI
@register_encoder_config("shufflenet_v2", IMAGE)
@ludwig_dataclass
class TVShuffleNetV2EncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("shufflenet_v2", description="Type of encoder.")

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
@register_encoder_config("squeezenet", IMAGE)
@ludwig_dataclass
class TVSqueezeNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("squeezenet", description="Type of encoder.")

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
@register_encoder_config("swin_transformer", IMAGE)
@ludwig_dataclass
class TVSwinTransformerEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("swin_transformer", description="Type of encoder.")

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
@register_encoder_config("vit", IMAGE)
@ludwig_dataclass
class TVViTEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("vit", description="Type of encoder.")

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
@register_encoder_config("vgg", IMAGE)
@ludwig_dataclass
class TVVGGEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("vgg", description="Type of encoder.")

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
@register_encoder_config("wide_resnet", IMAGE)
@ludwig_dataclass
class TVWideResNetEncoderConfig(TVBaseEncoderConfig):
    type: str = schema_utils.ProtectedString("wide_resnet", description="Type of encoder.")

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
