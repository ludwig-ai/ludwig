from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, BFILL, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_preprocessor(AUDIO)
@ludwig_dataclass
class AudioPreprocessingConfig(BasePreprocessingConfig):
    audio_file_length_limit_in_s: int = schema_utils.NonNegativeFloat(
        default=7.5,
        allow_none=False,
        description="Float value that defines the maximum limit of the audio file in seconds. All files longer than "
        "this limit are cut off. All files shorter than this limit are padded with padding_value",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["audio_file_length_limit_in_s"],
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=BFILL,
        allow_none=False,
        description="What strategy to follow when there's a missing value in an audio column",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["computed_fill_value"],
    )

    in_memory: bool = schema_utils.Boolean(
        default=True,
        description="Defines whether the audio dataset will reside in memory during the training process or will be "
        "dynamically fetched from disk (useful for large datasets). In the latter case a training batch "
        "of input audio will be fetched from disk each training iteration.",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["in_memory"],
    )

    padding_value: float = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="Float value that is used for padding.",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["padding_value"],
    )

    norm: str = schema_utils.StringOptions(
        ["per_file"],
        default=None,
        allow_none=True,
        description="Normalization strategy for the audio files. If None, no normalization is performed. If "
        "per_file, z-norm is applied on a 'per file' level",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["norm"],
    )

    type: str = schema_utils.StringOptions(
        ["fbank", "group_delay", "raw", "stft", "stft_phase"],
        default="fbank",
        description="Defines the type of audio feature to be used.",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["type"],
    )

    window_length_in_s: float = schema_utils.NonNegativeFloat(
        default=0.04,
        description="Defines the window length used for the short time Fourier transformation. This is only needed if "
        "the audio_feature_type is 'raw'.",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["window_length_in_s"],
    )

    window_shift_in_s: float = schema_utils.NonNegativeFloat(
        default=0.02,
        description="Defines the window shift used for the short time Fourier transformation (also called "
        "hop_length). This is only needed if the audio_feature_type is 'raw'. ",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["window_shift_in_s"],
    )

    num_fft_points: float = schema_utils.NonNegativeFloat(
        default=None,
        allow_none=True,
        description="Defines the number of fft points used for the short time Fourier transformation",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["num_fft_points"],
    )

    window_type: str = schema_utils.StringOptions(
        ["bartlett", "blackman", "hamming", "hann"],
        default="hamming",
        description="Defines the type window the signal is weighted before the short time Fourier transformation.",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["window_type"],
    )

    num_filter_bands: int = schema_utils.PositiveInteger(
        default=80,
        description="Defines the number of filters used in the filterbank. Only needed if audio_feature_type "
        "is 'fbank'",
        parameter_metadata=FEATURE_METADATA[AUDIO][PREPROCESSING]["num_filter_bands"],
    )
