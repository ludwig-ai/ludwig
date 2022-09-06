from marshmallow_dataclass import dataclass

from ludwig.constants import BFILL, IMAGE, MISSING_VALUE_STRATEGY_OPTIONS
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA


@register_preprocessor(IMAGE)
@dataclass
class ImagePreprocessingConfig(BasePreprocessingConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=BFILL,
        allow_none=False,
        description="What strategy to follow when there's a missing value in an image column",
    )

    fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )

    computed_fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    height: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The image height in pixels. If this parameter is set, images will be resized to the specified "
        "height using the resize_method parameter. If None, images will be resized to the size of the "
        "first image in the dataset.",
    )

    width: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The image width in pixels. If this parameter is set, images will be resized to the specified "
        "width using the resize_method parameter. If None, images will be resized to the size of the "
        "first image in the dataset.",
    )

    num_channels: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of channels in the images. If specified, images will be read in the mode specified by the "
        "number of channels. If not specified, the number of channels will be inferred from the image "
        "format of the first valid image in the dataset.",
    )

    resize_method: str = schema_utils.StringOptions(
        ["crop_or_pad", "interpolate"],
        default="interpolate",
        allow_none=False,
        description="The method to use for resizing images.",
    )

    infer_image_num_channels: bool = schema_utils.Boolean(
        default=True,
        description="If true, then the number of channels in the dataset is inferred from a sample of the first image "
        "in the dataset.",
    )

    infer_image_dimensions: bool = schema_utils.Boolean(
        default=True,
        description="If true, then the height and width of images in the dataset will be inferred from a sample of "
        "the first image in the dataset. Each image that doesn't conform to these dimensions will be "
        "resized according to resize_method. If set to false, then the height and width of images in the "
        "dataset will be specified by the user.",
    )

    infer_image_max_height: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="If infer_image_dimensions is set, this is used as the maximum height of the images in "
        "the dataset.",
    )

    infer_image_max_width: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="If infer_image_dimensions is set, this is used as the maximum width of the images in "
        "the dataset.",
    )

    infer_image_sample_size: int = schema_utils.PositiveInteger(
        default=100,
        allow_none=False,
        description="The sample size used for inferring dimensions of images in infer_image_dimensions.",
    )

    scaling: str = schema_utils.StringOptions(
        ["pixel_normalization", "pixel_standardization"],
        default="pixel_normalization",
        allow_none=False,
        description="The scaling strategy for pixel values in the image.",
    )

    in_memory: bool = schema_utils.Boolean(
        default=True,
        description="Defines whether image dataset will reside in memory during the training process or will be "
        "dynamically fetched from disk (useful for large datasets). In the latter case a training batch "
        "of input images will be fetched from disk each training iteration.",
    )

    num_processes: int = schema_utils.PositiveInteger(
        default=1,
        allow_none=False,
        description="Specifies the number of processes to run for preprocessing images.",
    )
