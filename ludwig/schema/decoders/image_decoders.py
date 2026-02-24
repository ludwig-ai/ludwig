from typing import Optional, TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.metadata import DECODER_METADATA
from ludwig.schema.utils import ludwig_dataclass

if TYPE_CHECKING:
    from ludwig.schema.features.preprocessing.image import ImagePreprocessingConfig


class ImageDecoderConfig(BaseDecoderConfig):
    def set_fixed_preprocessing_params(self, model_type: str, preprocessing: "ImagePreprocessingConfig"):
        preprocessing.requires_equal_dimensions = False
        preprocessing.height = None
        preprocessing.width = None


@DeveloperAPI
@register_decoder_config("unet", [IMAGE], model_types=[MODEL_ECD])
@ludwig_dataclass
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
