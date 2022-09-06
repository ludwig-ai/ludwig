from marshmallow_dataclass import dataclass

from ludwig.constants import AUDIO, BFILL, MISSING_VALUE_STRATEGY_OPTIONS
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA


@register_preprocessor(AUDIO)
@dataclass
class AudioPreprocessingConfig(BasePreprocessingConfig):

    audio_file_length_limit_in_s: int = schema_utils.NonNegativeFloat(
        default=7.5,
        allow_none=False,
        description="Float value that defines the maximum limit of the audio file in seconds. All files longer than "
        "this limit are cut off. All files shorter than this limit are padded with padding_value",
    )

    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=BFILL,
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
