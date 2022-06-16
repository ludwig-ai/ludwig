from abc import ABC
from dataclasses import field
from typing import ClassVar, Optional, Union, Any

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

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
    from the corresponding input feature class are copied over: check each class to check which attributes are
    different from the preprocessing of each feature.
    """

    feature_type: ClassVar[Optional[str]] = None
    "Class variable pointing to the corresponding preprocessor."

    type: str


@dataclass
@register_preprocessor("text")
class TextPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """TextPreprocessingConfig is a dataclass that configures the parameters used for a text input feature."""

    tokenizer: Optional[str] = schema_utils.String(
        default='space_punct',
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    vocab_file: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each line the "
                    "first string until \t or \n is considered a word.",
    )

    max_sequence_length: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
                    "truncated, while texts that are shorter will be padded.",
    )

    most_common: Optional[int] = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
                    "amount, the most infrequent symbols will be treated as unknown.",
    )

    padding_symbol: Optional[str] = schema_utils.String(
        default="<PAD>",
        allow_none=False,
        description="The string used as a padding symbol. This special token is mapped to the integer ID 0 in the "
                    "vocabulary.",
    )

    unknown_symbol: Optional[str] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The string used as an unknown placeholder. This special token is mapped to the integer ID 1 in "
                    "the vocabulary.",
    )

    padding: Optional[str] = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: Optional[str] = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace the missing values with in case the missing_value_strategy is fill_value",
    )


@dataclass
@register_preprocessor("number")
class NumberPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """NumberPreprocessingConfig is a dataclass that configures the parameters used for a number input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number column",
    )

    fill_value: Optional[float] = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    normalization: Optional[str] = schema_utils.StringOptions(
        ["zscore", "minmax", "log1p"],
        default=None,
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
    )


@dataclass
@register_preprocessor("binary")
class BinaryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """BinaryPreprocessingConfig is a dataclass that configures the parameters used for a binary input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_false", "fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_false",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a binary column",
    )

    fill_value: Union[int, float] = schema_utils.NumericOrStringOptionsField(
        ["yes", "YES", "Yes", "y", "Y", "true", "True", "TRUE", "t", "T", "1", "1.0", "no", "NO", "No", "n", "N",
         "false", "False", "FALSE", "f", "F", "0", "0.0"],
        allow_none=False,
        default=None,
        default_numeric=0,
        min=0,
        max=1,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    fallback_true_label: Optional[str] = schema_utils.NumericOrStringOptionsField(
        ["True", "False"],
        allow_none=True,
        default=None,
        default_numeric=1,
        default_option=None,
        min=0,
        max=1,
        description="The label to interpret as 1 (True) when the binary feature doesn't have a "
                    "conventional boolean value"
    )


@dataclass
@register_preprocessor("category")
class CategoryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """CategoryPreprocessingConfig is a dataclass that configures the parameters used for a category input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category column",
    )

    fill_value: Optional[str] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
    )

    most_common_label: Optional[int] = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
                    "amount, the most infrequent tokens will be treated as unknown.",
    )


@dataclass
@register_preprocessor("set")
class SetPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: Optional[str] = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
                    "default value space splits the string on spaces. Common options include: underscore (splits on "
                    "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
                    "JSON parser).",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
    )

    fill_value: Optional[Any] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    most_common_label: Optional[int] = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
                    "amount, the most infrequent tokens will be treated as unknown.",
    )


@dataclass
@register_preprocessor("sequence")
class SequencePreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: Optional[str] = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    vocab_file: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each line the "
                    "first string until \t or \n is considered a word.",
    )

    max_sequence_length: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
                    "truncated, while texts that are shorter will be padded.",
    )

    most_common: Optional[int] = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
                    "amount, the most infrequent symbols will be treated as unknown.",
    )

    padding_symbol: Optional[str] = schema_utils.String(
        default="<PAD>",
        allow_none=False,
        description="The string used as a padding symbol. This special token is mapped to the integer ID 0 in the "
                    "vocabulary.",
    )

    unknown_symbol: Optional[str] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The string used as an unknown placeholder. This special token is mapped to the integer ID 1 in "
                    "the vocabulary.",
    )

    padding: Optional[str] = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: Optional[str] = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace the missing values with in case the missing_value_strategy is fill_value",
    )


@dataclass
@register_preprocessor("image")
class ImagePreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    height: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The image height in pixels. If this parameter is set, images will be resized to the specified "
                    "height using the resize_method parameter. If None, images will be resized to the size of the "
                    "first image in the dataset.",
    )

    width: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The image width in pixels. If this parameter is set, images will be resized to the specified "
                    "width using the resize_method parameter. If None, images will be resized to the size of the "
                    "first image in the dataset.",
    )

    num_channels: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Number of channels in the images. If specified, images will be read in the mode specified by the "
                    "number of channels. If not specified, the number of channels will be inferred from the image "
                    "format of the first valid image in the dataset.",
    )

    resize_method: Optional[str] = schema_utils.StringOptions(
        ["crop_or_pad", "interpolate"],
        default="crop_or_pad",
        allow_none=False,
        description="The method to use for resizing images.",
    )

    infer_image_num_channels: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="If true, then the number of channels in the dataset is inferred from a sample of the first image "
                    "in the dataset.",
    )

    infer_image_dimensions: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="If true, then the height and width of images in the dataset will be inferred from a sample of "
                    "the first image in the dataset. Each image that doesn't conform to these dimensions will be "
                    "resized according to resize_method. If set to false, then the height and width of images in the "
                    "dataset will be specified by the user.",
    )

    infer_image_max_height: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="If infer_image_dimensions is set, this is used as the maximum height of the images in "
                    "the dataset.",
    )

    infer_image_max_width: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="If infer_image_dimensions is set, this is used as the maximum width of the images in "
                    "the dataset.",
    )

    infer_image_sample_size: Optional[int] = schema_utils.PositiveInteger(
        default=100,
        allow_none=False,
        description="The sample size used for inferring dimensions of images in infer_image_dimensions.",
    )

    scaling_method: Optional[str] = schema_utils.StringOptions(
        ["pixel_normalization", "pixel_standardization"],
        default="pixel_normalization",
        allow_none=False,
        description="The scaling strategy for pixel values in the image.",
    )

    in_memory: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Defines whether image dataset will reside in memory during the training process or will be "
                    "dynamically fetched from disk (useful for large datasets). In the latter case a training batch "
                    "of input images will be fetched from disk each training iteration.",
    )

    num_processes: Optional[int] = schema_utils.PositiveInteger(
        default=1,
        allow_none=False,
        description="Specifies the number of processes to run for preprocessing images.",
    )


@dataclass
@register_preprocessor("audio")
class AudioPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    audio_file_length_limit_in_s: Optional[int] = schema_utils.NonNegativeFloat(
        default=7.5,
        allow_none=False,
        description="Float value that defines the maximum limit of the audio file in seconds. All files longer than "
                    "this limit are cut off. All files shorter than this limit are padded with padding_value",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="backfill",
        allow_none=False,
        description="What strategy to follow when there's a missing value in an audio column",
    )

    in_memory: Optional[bool] = schema_utils.Boolean(
        default=True,
        description="Defines whether the audio dataset will reside in memory during the training process or will be "
                    "dynamically fetched from disk (useful for large datasets). In the latter case a training batch "
                    "of input audio will be fetched from disk each training iteration.",
    )

    padding_value: Optional[float] = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="Float value that is used for padding."
    )

    norm: Optional[str] = schema_utils.StringOptions(
        ["per_file"],
        default=None,
        allow_none=True,
        description="Normalization strategy for the audio files. If None, no normalization is performed. If "
                    "per_file, z-norm is applied on a 'per file' level",
    )

    audio_feature: Optional[dict] = schema_utils.Dict(
        default={"type": "raw"},
        description="Dictionary that takes as input the audio feature type as well as additional parameters if type "
                    "!= raw. The following parameters can/should be defined in the dictionary "
    )

    type: Optional[str] = schema_utils.StringOptions(
        ["raw", "stft", "stft_phase", "group_delay"],
        default="raw",
        allow_none=False,
        description="Defines the type of audio features to be used.",
    )

    window_length_in_s: Optional[float] = schema_utils.NonNegativeFloat(
        default=0.04,
        allow_none=False,
        description="Defines the window length used for the short time Fourier transformation (only needed if type is "
                    "not raw).",
    )

    window_shift_in_s: Optional[float] = schema_utils.NonNegativeFloat(
        default=0.02,
        allow_none=False,
        description="Defines the window shift used for the short time Fourier transformation - also called hop length "
                    "(only needed if type is not raw).",
    )

    num_fft_points: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=False,
        description="Defines the number of fft points used for the short time Fourier transformation. If "
                    "num_fft_points > (window_length_in_s * sample_rate), then the signal is zero-padded at the end. "
                    "num_fft_points has to be >= (window_length_in_s * sample_rate). Only needed if type is not raw.",
    )

    window_type: Optional[str] = schema_utils.String(
        default="hamming",
        allow_none=False,
        description="Defines the window type the signal is weighted before the short time Fourier transformation. All "
                    "windows provided by scipy’s window function can be used (only needed if type != raw).",
        )

    num_filter_bands: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=False,
        description="Defines the number of filters used in the filterbank (only needed if type == fbank)",
    )


@dataclass
@register_preprocessor("timeseries")
class TimeseriesPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: Optional[str] = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to map from the raw string content of the dataset column to a sequence of elements.",
    )

    vocab_file: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Filepath string to a UTF-8 encoded file containing the sequence's vocabulary. On each line the "
                    "first string until \t or \n is considered a word.",
    )

    max_sequence_length: Optional[int] = schema_utils.PositiveInteger(
        default=256,
        allow_none=False,
        description="The maximum length (number of tokens) of the text. Texts that are longer than this value will be "
                    "truncated, while texts that are shorter will be padded.",
    )

    most_common: Optional[int] = schema_utils.PositiveInteger(
        default=20000,
        allow_none=False,
        description="The maximum number of most common tokens in the vocabulary. If the data contains more than this "
                    "amount, the most infrequent symbols will be treated as unknown.",
    )

    padding_symbol: Optional[str] = schema_utils.String(
        default="<PAD>",
        allow_none=False,
        description="The string used as a padding symbol. This special token is mapped to the integer ID 0 in the "
                    "vocabulary.",
    )

    unknown_symbol: Optional[str] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The string used as an unknown placeholder. This special token is mapped to the integer ID 1 in "
                    "the vocabulary.",
    )

    padding: Optional[str] = schema_utils.StringOptions(
        ["left", "right"],
        default="right",
        allow_none=False,
        description="the direction of the padding. right and left are available options.",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a text column",
    )

    fill_value: Optional[str] = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace the missing values with in case the missing_value_strategy is fill_value",
    )


@dataclass
@register_preprocessor("bag")
class BagPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    tokenizer: Optional[str] = schema_utils.String(
        default="space",
        allow_none=False,
        description="Defines how to transform the raw text content of the dataset column to a set of elements. The "
                    "default value space splits the string on spaces. Common options include: underscore (splits on "
                    "underscore), comma (splits on comma), json (decodes the string into a set or a list through a "
                    "JSON parser).",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a set column",
    )

    fill_value: Optional[Any] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="If true, converts the string to lowercase before tokenizing.",
    )

    most_common_label: Optional[int] = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. If the data contains more than this "
                    "amount, the most infrequent tokens will be treated as unknown.",
    )


@dataclass
@register_preprocessor("h3")
class H3PreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in an h3 column",
    )

    fill_value: Optional[Any] = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )


@dataclass
@register_preprocessor("date")
class DatePreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a date column",
    )

    fill_value: Optional[Any] = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    datetime_format: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="This parameter can either be a datetime format string, or null, in which case the datetime "
                    "format will be inferred automatically.",
    )


@dataclass
@register_preprocessor("vector")
class VectorPreprocessingConfig(schema_utils.BaseMarshmallowConfig):

    vector_size: Optional[int] = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The size of the vector. If None, the vector size will be inferred from the data.",
    )

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a vector column",
    )

    # TODO (Connor): Add optional pattern arg for string input
    fill_value: Optional[Any] = schema_utils.String(
        default="",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )


def PreprocessingDataclassField(feature_type: str):
    """
    Custom dataclass field that when used inside a dataclass will allow the user to specify a preprocessing config.

    Returns: Inialized dataclass field that converts an untyped dict with params to a preprocessing config.
    """

    class PreprocessingMarshmallowField(fields.Field):
        """
        Custom marshmallow field that deserializes a dict for a valid preprocessing config from the
        preprocessing_registry and creates a corresponding `oneOf` JSON schema for external usage.
        """

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if "type" in value and value["type"] in preprocessing_registry:
                    opt = preprocessing_registry[value["type"].lower()][1]
                    try:
                        return opt.Schema().load(value)
                    except (TypeError, ValidationError) as e:
                        raise ValidationError(
                            f"Invalid params for optimizer: {value}, see `{opt}` definition. Error: {e}"
                        )
                raise ValidationError(
                    f"Invalid params for optimizer: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": preprocessing_registry[feature_type].Schema(),
                "additionalProperties": False,
            }

    try:
        preprocessor = preprocessing_registry[feature_type]
        load_default = preprocessor.Schema().load(default)
        dump_default = preprocessor.Schema().dump(default)

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
        raise ValidationError(f"Unsupported preprocessing type: {feature_type}. See optimizer_registry. Details: {e}")