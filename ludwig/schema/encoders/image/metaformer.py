from typing import Optional

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.image.base import ImageEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.utils import ludwig_dataclass

@DeveloperAPI
@register_encoder_config("metaformer", IMAGE)
@ludwig_dataclass
class MetaFormerConfig(ImageEncoderConfig):
    """Configuration for the MetaFormer / CAFormer style image encoder.

    This schema intentionally avoids referencing ENCODER_METADATA (not yet extended)
    to keep the initial integration minimal and self-contained.
    """

    @staticmethod
    def module_name():
        return "MetaFormerEncoder"

    type: str = schema_utils.ProtectedString(
        "metaformer",
        description="MetaFormer / CAFormer image encoder integrating ConvFormer / CAFormer style backbones.",
    )

    model_name: str = schema_utils.String(
        default="caformer_s18",
        allow_none=False,
        description="Backbone model name (e.g. caformer_s18, caformer_s36, caformer_m36, caformer_b36, etc.).",
    )

    use_pretrained: bool = schema_utils.Boolean(
        default=True,
        description="If true, load pretrained backbone weights (if available).",
    )

    trainable: bool = schema_utils.Boolean(
        default=True,
        description="If false, freezes backbone parameters.",
    )

    output_size: int = schema_utils.PositiveInteger(
        default=128,
        description="Projection head output dimensionality.",
    )

    height: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Input image height (optional; if None, provided by feature preprocessing).",
    )

    width: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Input image width (optional; if None, provided by feature preprocessing).",
    )

    num_channels: Optional[int] = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of input image channels (e.g. 1 for grayscale, 3 for RGB).",
    )

    def set_fixed_preprocessing_params(self, model_type: str, preprocessing: "ImagePreprocessingConfig"):
        # Allow variable sizes; internal wrapper adapts / pools to model expected size.
        preprocessing.requires_equal_dimensions = False
        # Leave height/width unset to allow dataset-driven or on-the-fly resizing.
        # Channel adaptation is handled dynamically if needed.
        if self.height is not None:
            preprocessing.height = self.height
        if self.width is not None:
            preprocessing.width = self.width
