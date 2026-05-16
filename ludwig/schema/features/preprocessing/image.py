from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BFILL, DROP_ROW, IMAGE, IMAGENET1K, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA


@DeveloperAPI
@register_preprocessor(IMAGE)
class ImagePreprocessingConfig(BasePreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=BFILL,
        allow_none=False,
        description="What strategy to follow when there's a missing value in an image column",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["computed_fill_value"],
    )

    height: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The image height in pixels. If this parameter is set, images will be resized to the specified "
        "height using the resize_method parameter. If None, images will be resized to the size of the "
        "first image in the dataset.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["height"],
    )

    width: int | None = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The image width in pixels. If this parameter is set, images will be resized to the specified "
        "width using the resize_method parameter. If None, images will be resized to the size of the "
        "first image in the dataset.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["width"],
    )

    num_channels: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of channels in the images. If specified, images will be read in the mode specified by the "
        "number of channels. If not specified, the number of channels will be inferred from the image "
        "format of the first valid image in the dataset.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["num_channels"],
    )

    resize_method: str = schema_utils.StringOptions(
        ["crop_or_pad", "interpolate"],
        default="interpolate",
        allow_none=False,
        description="The method to use for resizing images.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["resize_method"],
    )

    infer_image_num_channels: bool = schema_utils.Boolean(
        default=True,
        description="If true, then the number of channels in the dataset is inferred from a sample of the first image "
        "in the dataset.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["infer_image_num_channels"],
    )

    infer_image_dimensions: bool = schema_utils.Boolean(
        default=True,
        description="If true, then the height and width of images in the dataset will be inferred from a sample of "
        "the first image in the dataset. Each image that doesn't conform to these dimensions will be "
        "resized according to resize_method. If set to false, then the height and width of images in the "
        "dataset will be specified by the user.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["infer_image_dimensions"],
    )

    infer_image_max_height: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="If infer_image_dimensions is set, this is used as the maximum height of the images in "
        "the dataset.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["infer_image_max_height"],
    )

    infer_image_max_width: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="If infer_image_dimensions is set, this is used as the maximum width of the images in the dataset.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["infer_image_max_width"],
    )

    infer_image_sample_size: int = schema_utils.PositiveInteger(
        default=100,
        allow_none=False,
        description="The sample size used for inferring dimensions of images in infer_image_dimensions.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["infer_image_sample_size"],
    )

    standardize_image: str | None = schema_utils.StringOptions(
        [IMAGENET1K],
        default=None,
        allow_none=True,
        description="Standardize image by per channel mean centering and standard deviation scaling .",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["standardize_image"],
    )

    in_memory: bool = schema_utils.Boolean(
        default=True,
        description="Defines whether image dataset will reside in memory during the training process or will be "
        "dynamically fetched from disk (useful for large datasets). In the latter case a training batch "
        "of input images will be fetched from disk each training iteration.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["in_memory"],
    )

    num_processes: int = schema_utils.PositiveInteger(
        default=1,
        allow_none=False,
        description="Specifies the number of processes to run for preprocessing images.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["num_processes"],
    )

    requires_equal_dimensions: bool = schema_utils.Boolean(
        default=False,
        description="If true, then width and height must be equal.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["requires_equal_dimensions"],
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of channel classes in the images. If specified, this value will be validated "
        "against the inferred number of classes. Use 2 to convert grayscale images to binary images.",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["num_classes"],
    )

    infer_image_num_classes: bool = schema_utils.Boolean(
        default=False,
        description="If true, then the number of channel classes in the dataset will be inferred from a sample of "
        "the first image in the dataset. Each unique channel value will be mapped to a class and preprocessing will "
        "create a masked image based on the channel classes. ",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["infer_image_num_classes"],
    )

    mode: str = schema_utils.StringOptions(
        ["eager", "lazy", "lazy_cached"],
        default="lazy",
        allow_none=False,
        description="Preprocessing mode for image features. 'eager' decodes all files during preprocessing "
        "and stores tensors in the Parquet cache. 'lazy' stores file paths and decodes per batch during "
        "training, keeping peak memory bounded to batch_size × image_size. 'lazy_cached' behaves like "
        "'lazy' on the first training epoch but writes decoded tensors to a numpy memmap alongside the "
        "Parquet cache; subsequent epochs read from the memmap directly, eliminating decode overhead. "
        "Lazy mode is disabled automatically when a torchvision pretrained encoder is used.",
    )

    prefetch_size: int | None = schema_utils.NonNegativeInteger(
        default=None,
        allow_none=True,
        description="Number of batches to prefetch in a background thread while the GPU processes the current "
        "batch. None (default) selects automatically: 0 for 'eager' mode, 4 for 'lazy' and "
        "'lazy_cached' (epoch 1). After the first epoch in 'lazy_cached' mode, prefetch is "
        "automatically disabled since memmap reads are fast enough. Set to 0 to disable prefetch "
        "entirely, or to a positive integer to override the automatic selection.",
    )

    lazy_cache_dir: str | None = schema_utils.String(
        default=None,
        allow_none=True,
        description="Directory in which to cache image files when the source data is in-memory (e.g. a "
        "HuggingFace dataset). Only used when mode is 'lazy' or 'lazy_cached' and the input entries "
        "are not already paths to existing files. When None, defaults to "
        "~/.cache/ludwig/lazy_media/<feature_name>/. Has no effect when the input column already "
        "contains local file paths.",
    )


@DeveloperAPI
@register_preprocessor("image_output")
class ImageOutputPreprocessingConfig(ImagePreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=DROP_ROW,
        allow_none=False,
        description="What strategy to follow when there's a missing value in an image column",
        parameter_metadata=FEATURE_METADATA[IMAGE][PREPROCESSING]["missing_value_strategy"],
    )
