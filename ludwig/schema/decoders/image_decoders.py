from typing import TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA

if TYPE_CHECKING:
    from ludwig.schema.features.preprocessing.image import ImagePreprocessingConfig


class ImageDecoderConfig(BaseDecoderConfig):
    def set_fixed_preprocessing_params(self, model_type: str, preprocessing: "ImagePreprocessingConfig"):
        preprocessing.requires_equal_dimensions = False
        preprocessing.height = None
        preprocessing.width = None


@DeveloperAPI
@register_decoder_config("unet", [IMAGE], model_types=[MODEL_ECD])
class UNetDecoderConfig(ImageDecoderConfig):
    @staticmethod
    def module_name():
        return "UNetDecoder"

    type: str = schema_utils.ProtectedString(
        "unet",
        description=DECODER_METADATA["UNetDecoder"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=1024,
        description="Size of the input to the decoder.",
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["input_size"],
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the output image.",
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the output image.",
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["width"],
    )

    num_channels: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of channels in the output image. ",
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["num_channels"],
    )

    conv_norm: str | None = schema_utils.StringOptions(
        ["batch"],
        default="batch",
        allow_none=True,
        description="This is the default norm that will be used for each double conv layer." "It can be null or batch.",
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["conv_norm"],
    )

    num_classes: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of classes to predict in the output. ",
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["num_classes"],
    )

    num_stages: int = schema_utils.PositiveInteger(
        default=4,
        description=(
            "Number of encoder/decoder stage pairs in the UNet. "
            "The input image dimensions must be divisible by 2^num_stages. "
            "Increasing this value lets the model capture features at more spatial scales."
        ),
        parameter_metadata=DECODER_METADATA["UNetDecoder"]["num_stages"],
    )


@DeveloperAPI
@register_decoder_config("segformer", [IMAGE], model_types=[MODEL_ECD])
class SegFormerDecoderConfig(ImageDecoderConfig):
    """Config for the SegFormer MLP decoder head.

    Reference: Xie et al., "SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers", NeurIPS 2021.
    https://arxiv.org/abs/2105.15203
    """

    @staticmethod
    def module_name():
        return "SegFormerDecoder"

    type: str = schema_utils.ProtectedString(
        "segformer",
        description=DECODER_METADATA["SegFormerDecoder"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input feature vector from the combiner.",
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["input_size"],
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the output segmentation map.",
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the output segmentation map.",
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["width"],
    )

    num_channels: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of channels in the input image (informational; set from preprocessing).",
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["num_channels"],
    )

    num_classes: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of segmentation classes to predict.",
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["num_classes"],
    )

    hidden_size: int = schema_utils.PositiveInteger(
        default=256,
        description=(
            "Width of the hidden MLP projection applied to the feature map before upsampling. "
            "Larger values increase capacity but also compute cost."
        ),
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["hidden_size"],
    )

    dropout: float = schema_utils.FloatRange(
        default=0.1,
        min=0.0,
        max=1.0,
        description="Dropout probability applied after the hidden MLP projection.",
        parameter_metadata=DECODER_METADATA["SegFormerDecoder"]["dropout"],
    )


@DeveloperAPI
@register_decoder_config("fpn", [IMAGE], model_types=[MODEL_ECD])
class FPNDecoderConfig(ImageDecoderConfig):
    """Config for the Feature Pyramid Network (FPN) decoder.

    Reference: Lin et al., "Feature Pyramid Networks for Object Detection",
    CVPR 2017. https://arxiv.org/abs/1612.03144
    """

    @staticmethod
    def module_name():
        return "FPNDecoder"

    type: str = schema_utils.ProtectedString(
        "fpn",
        description=DECODER_METADATA["FPNDecoder"]["type"].long_description,
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input feature vector from the combiner.",
        parameter_metadata=DECODER_METADATA["FPNDecoder"]["input_size"],
    )

    height: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Height of the output segmentation map.",
        parameter_metadata=DECODER_METADATA["FPNDecoder"]["height"],
    )

    width: int = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Width of the output segmentation map.",
        parameter_metadata=DECODER_METADATA["FPNDecoder"]["width"],
    )

    num_classes: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of segmentation classes to predict.",
        parameter_metadata=DECODER_METADATA["FPNDecoder"]["num_classes"],
    )

    num_channels: int = schema_utils.PositiveInteger(
        default=256,
        description=(
            "Number of channels in each FPN level after the lateral 1x1 projection. "
            "All pyramid levels are projected to this width before the top-down merge."
        ),
        parameter_metadata=DECODER_METADATA["FPNDecoder"]["num_channels"],
    )

    num_levels: int = schema_utils.PositiveInteger(
        default=4,
        description=(
            "Number of pyramid levels to build in the top-down pathway. "
            "More levels capture coarser context; typical range is 2-5."
        ),
        parameter_metadata=DECODER_METADATA["FPNDecoder"]["num_levels"],
    )
