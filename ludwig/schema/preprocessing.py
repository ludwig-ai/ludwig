from abc import ABC
from dataclasses import field
from typing import ClassVar, Optional, Union

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.constants import (
    AUDIO,
    BACKFILL,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.utils import strings_utils
from ludwig.utils.registry import Registry
from ludwig.utils.tokenizers import tokenizer_registry

preprocessing_registry = Registry()


def register_preprocessor(name: str):
    def wrap(preprocessing_config: BasePreprocessingConfig):
        preprocessing_registry[name] = preprocessing_config
        return preprocessing_config

    return wrap


@dataclass
class BasePreprocessingConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for input feature preprocessing. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding input feature class are copied over: check each class to check which attributes are different
    from the preprocessing of each feature.
    """

    feature_type: ClassVar[Optional[str]] = None
    "Class variable pointing to the corresponding preprocessor."

    type: str


@register_preprocessor(TEXT)
@dataclass
class TextPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """TextPreprocessingConfig is a dataclass that configures the parameters used for a text input feature."""

    pretrained_model_name_or_path: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="This can be either the name of a pretrained HuggingFace model or a path where it was downloaded",
    )

    tokenizer: str = schema_utils.StringOptions(
        tokenizer_registry.keys(),
        default="space_punct",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    vocab_file: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each line the "
        "first string until \t or \n is considered a word.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
        "truncated, while texts that are shorter will be padded.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
        "amount, the most infrequent symbols will be treated as unknown.",
    )

    padding_symbol: str = schema_utils.String(
        default="<PAD>",
        allow_none=False,
        description="The string used as the padding symbol for sequence features. Ignored for features using "
        "huggingface encoders, which have their own vocabulary.",
    )

    unknown_symbol: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The string used as the unknown symbol for sequence features. Ignored for features using "
        "huggingface encoders, which have their own vocabulary.",
    )

    padding: str = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    lowercase: bool = schema_utils.Boolean(
        default=True,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    computed_fill_value: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="The computed fill value determined by the user or inferred from the data.",
    )


@register_preprocessor(NUMBER)
@dataclass
class NumberPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """NumberPreprocessingConfig is a dataclass that configures the parameters used for a number input feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number column",
    )

    fill_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    normalization: str = schema_utils.StringOptions(
        ["zscore", "minmax", "log1p"],
        default=None,
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
    )


@register_preprocessor(BINARY)
@dataclass
@register_preprocessor(BINARY)
class BinaryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """BinaryPreprocessingConfig is a dataclass that configures the parameters used for a binary input feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS + ["fill_with_false"],
        default="fill_with_false",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a binary column",
    )

    fill_value: Union[int, float, str] = schema_utils.NumericOrStringOptionsField(
        strings_utils.all_bool_strs(),
        default=None,
        default_numeric=None,
        default_option=None,
        allow_none=False,
        min=0,
        max=1,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: Union[int, float, str] = schema_utils.NumericOrStringOptionsField(
        strings_utils.all_bool_strs(),
        default=None,
        default_numeric=None,
        default_option=None,
        allow_none=False,
        min=0,
        max=1,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    fallback_true_label: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The label to interpret as 1 (True) when the binary feature doesn't have a "
        "conventional boolean value",
    )


@register_preprocessor(CATEGORY)
@dataclass
class CategoryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """CategoryPreprocessingConfig is a dataclass that configures the parameters used for a category input
    feature."""

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )


@register_preprocessor(SET)
@dataclass
class SetPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
        "default value space splits the string on spaces. Common options include: underscore (splits on "
        "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
        "JSON parser).",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )


@register_preprocessor(SEQUENCE)
@dataclass
class SequencePreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: str = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    vocab_file: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each line the "
        "first string until \t or \n is considered a word.",
    )

    max_sequence_length: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
        "truncated, while texts that are shorter will be padded.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
        "amount, the most infrequent symbols will be treated as unknown.",
    )

    padding_symbol: str = schema_utils.String(
        default="<PAD>",
        allow_none=False,
        description="The string used as a padding symbol. This special token is mapped to the integer ID 0 in the "
        "vocabulary.",
    )

    unknown_symbol: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The string used as an unknown placeholder. This special token is mapped to the integer ID 1 in "
        "the vocabulary.",
    )

    padding: str = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    computed_fill_value: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="The computed fill value determined by the user or inferred from the data.",
    )


@register_preprocessor(IMAGE)
@dataclass
class ImagePreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="backfill",
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


@register_preprocessor(AUDIO)
@dataclass
class AudioPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    audio_file_length_limit_in_s: int = schema_utils.NonNegativeFloat(
        default=7.5,
        allow_none=False,
        description="Float value that defines the maximum limit of the audio file in seconds. All files longer than "
        "this limit are cut off. All files shorter than this limit are padded with padding_value",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=BACKFILL,
        allow_none=False,
        description="What strategy to follow when there's a missing value in an audio column",
    )

    fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    in_memory: bool = schema_utils.Boolean(
        default=True,
        description="Defines whether the audio dataset will reside in memory during the training process or will be "
        "dynamically fetched from disk (useful for large datasets). In the latter case a training batch "
        "of input audio will be fetched from disk each training iteration.",
    )

    padding_value: float = schema_utils.NonNegativeFloat(
        default=0.0, allow_none=False, description="Float value that is used for padding."
    )

    norm: str = schema_utils.StringOptions(
        ["per_file"],
        default=None,
        allow_none=True,
        description="Normalization strategy for the audio files. If None, no normalization is performed. If "
        "per_file, z-norm is applied on a 'per file' level",
    )

    type: str = schema_utils.StringOptions(
        ["fbank", "group_delay", "raw", "stft", "stft_phase"],
        default="fbank",
        description="Defines the type of audio feature to be used.",
    )

    window_length_in_s: float = schema_utils.NonNegativeFloat(
        default=0.04,
        description="Defines the window length used for the short time Fourier transformation. This is only needed if "
        "the audio_feature_type is 'raw'.",
    )

    window_shift_in_s: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="Defines the window shift used for the short time Fourier transformation (also called "
        "hop_length). This is only needed if the audio_feature_type is 'raw'. ",
    )

    num_fft_points: float = schema_utils.NonNegativeFloat(
        default=None, description="Defines the number of fft points used for the short time Fourier transformation"
    )

    window_type: str = schema_utils.StringOptions(
        ["bartlett", "blackman", "hamming", "hann"],
        default="hamming",
        description="Defines the type window the signal is weighted before the short time Fourier transformation.",
    )

    num_filter_bands: int = schema_utils.PositiveInteger(
        default=80,
        description="Defines the number of filters used in the filterbank. Only needed if audio_feature_type "
        "is 'fbank'",
    )


@register_preprocessor(TIMESERIES)
@dataclass
class TimeseriesPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: str = schema_utils.StringOptions(
        sorted(list(tokenizer_registry.keys())),
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    timeseries_length_limit: int = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="Defines the maximum length of the timeseries. All timeseries longer than this limit are cut off.",
    )

    padding_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="Float value that is used for padding.",
    )

    padding: str = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )


@register_preprocessor(BAG)
@dataclass
class BagPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: str = schema_utils.StringOptions(
        tokenizer_registry.keys(),
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
        "default value space splits the string on spaces. Common options include: underscore (splits on "
        "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
        "JSON parser).",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
    )

    fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default=strings_utils.UNKNOWN_SYMBOL,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    lowercase: bool = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    most_common: int = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
        "amount, the most infrequent tokens will be treated as unknown.",
    )


@register_preprocessor(H3)
@dataclass
class H3PreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in an h3 column",
    )

    fill_value: int = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: int = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )


@register_preprocessor(DATE)
@dataclass
class DatePreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a date column",
    )

    fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )

    datetime_format: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="This parameter can either be a datetime format string, or null, in which case the datetime "
        "format will be inferred automatically.",
    )


@register_preprocessor(VECTOR)
@dataclass
class VectorPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    vector_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The size of the vector. If None, the vector size will be inferred from the data.",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a vector column",
    )

    fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        pattern=r"^([0-9]+(\.[0-9]*)?\s*)*$",
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    computed_fill_value: str = schema_utils.String(
        default="",
        allow_none=False,
        pattern=r"^([0-9]+(\.[0-9]*)?\s*)*$",
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=PREPROCESSING_METADATA["computed_fill_value"],
    )


def PreprocessingDataclassField(feature_type: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a preprocessing
    config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a preprocessing config.
    """

    class PreprocessingMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid preprocessing config from the
        preprocessing_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if feature_type in preprocessing_registry:
                    pre = preprocessing_registry[feature_type]
                    try:
                        return pre.Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid preprocessing params: {value}, see `{pre}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for preprocessor: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            preprocessor_cls = preprocessing_registry[feature_type]
            props = schema_utils.unload_jsonschema_from_marshmallow_class(preprocessor_cls)["properties"]
            return {
                "type": "object",
                "properties": props,
                "additionalProperties": False,
            }

    try:
        preprocessor = preprocessing_registry[feature_type]
        load_default = preprocessor.Schema().load({"feature_type": feature_type})
        dump_default = preprocessor.Schema().dump({"feature_type": feature_type})

        return field(
            metadata={
                "marshmallow_field": PreprocessingMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported preprocessing type: {feature_type}. See preprocessing_registry. " f"Details: {e}"
        )
