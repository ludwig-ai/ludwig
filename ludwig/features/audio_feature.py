#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from packaging import version

from ludwig.constants import AUDIO, AUDIO_FEATURE_KEYS, COLUMN, NAME, PREPROCESSING, PROC_COLUMN, SRC, TYPE
from ludwig.features.base_feature import BaseFeatureMixin
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.schema.features.audio_feature import AudioInputFeatureConfig
from ludwig.types import FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils.audio_utils import (
    calculate_mean,
    calculate_var,
    get_default_audio,
    get_fbank,
    get_group_delay,
    get_length_in_samp,
    get_max_length_stft_based,
    get_non_symmetric_length,
    get_phase_stft_magnitude,
    get_stft_magnitude,
    is_torch_audio_tuple,
    read_audio_from_bytes_obj,
    read_audio_from_path,
)
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import has_remote_protocol
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class _AudioPreprocessing(torch.nn.Module):
    audio_feature_dict: Dict[str, Union[float, int, str]]

    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.audio_feature_dict = {
            key: value
            for key, value in metadata["preprocessing"].items()
            if key in AUDIO_FEATURE_KEYS and value is not None
        }
        self.feature_dim = metadata["feature_dim"]
        self.max_length = metadata["max_length"]
        self.padding_value = metadata["preprocessing"]["padding_value"]
        self.normalization_type = metadata["preprocessing"]["norm"]

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if not torch.jit.isinstance(v, List[Tuple[torch.Tensor, int]]):
            raise ValueError(f"Unsupported input: {v}")

        processed_audio_matrix = []
        for audio, sampling_rate_in_hz in v:
            processed_audio = AudioFeatureMixin._transform_to_feature(
                audio,
                sampling_rate_in_hz,
                self.audio_feature_dict,
                self.feature_dim,
                self.max_length,
                self.padding_value,
                self.normalization_type,
            )
            processed_audio_matrix.append(processed_audio)
        return torch.stack(processed_audio_matrix)


class AudioFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return AUDIO

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        first_audio_file_path = column.head(1).iloc[0]
        _, sampling_rate_in_hz = torchaudio.load(first_audio_file_path)

        feature_dim = AudioFeatureMixin._get_feature_dim(preprocessing_parameters, sampling_rate_in_hz)
        audio_file_length_limit_in_s = preprocessing_parameters["audio_file_length_limit_in_s"]
        max_length = AudioFeatureMixin._get_max_length_feature(
            preprocessing_parameters, sampling_rate_in_hz, audio_file_length_limit_in_s
        )
        return {
            "feature_dim": feature_dim,
            "sampling_rate_in_hz": sampling_rate_in_hz,
            "max_length": max_length,
            "reshape": (max_length, feature_dim),
        }

    @staticmethod
    def _get_feature_dim(preprocessing_parameters: PreprocessingConfigDict, sampling_rate_in_hz):
        feature_type = preprocessing_parameters[TYPE]

        if feature_type == "raw":
            feature_dim = 1
        elif feature_type == "stft_phase":
            feature_dim_symmetric = get_length_in_samp(
                preprocessing_parameters["window_length_in_s"], sampling_rate_in_hz
            )
            feature_dim = 2 * get_non_symmetric_length(feature_dim_symmetric)
        elif feature_type in ["stft", "group_delay"]:
            feature_dim_symmetric = get_length_in_samp(
                preprocessing_parameters["window_length_in_s"], sampling_rate_in_hz
            )
            feature_dim = get_non_symmetric_length(feature_dim_symmetric)
        elif feature_type == "fbank":
            feature_dim = preprocessing_parameters["num_filter_bands"]
        else:
            raise ValueError(f"{feature_type} is not recognized.")

        return feature_dim

    @staticmethod
    def _process_in_memory(
        column,
        audio_feature_dict,
        feature_dim,
        max_length,
        padding_value,
        normalization_type,
        audio_file_length_limit_in_s,
        backend,
    ):
        df_engine = backend.df_engine
        if version.parse(torch.__version__) > version.parse("2.0.0"):
            # Read audio from path if the version of torch is > 2.0.0.
            raw_audio = backend.read_binary_files(column, map_fn=read_audio_from_path)
        else:
            raw_audio = backend.read_binary_files(column, map_fn=read_audio_from_bytes_obj)

        try:
            default_audio = get_default_audio([audio for audio in raw_audio if is_torch_audio_tuple(audio)])
        except RuntimeError:
            logger.info("Unable to process audio files provided")
            raise RuntimeError

        raw_audio = df_engine.map_objects(raw_audio, lambda row: row if is_torch_audio_tuple(row) else default_audio)
        processed_audio = df_engine.map_objects(
            raw_audio,
            lambda row: AudioFeatureMixin._transform_to_feature(
                audio=row[0],
                sampling_rate_in_hz=row[1],
                audio_feature_dict=audio_feature_dict,
                feature_dim=feature_dim,
                max_length=max_length,
                padding_value=padding_value,
                normalization_type=normalization_type,
            ).numpy(),  # non-torchscript preprocessing requires np.ndarray
        )

        audio_stats = df_engine.map_objects(
            raw_audio,
            lambda row: AudioFeatureMixin._get_stats(
                audio=row[0],
                sampling_rate_in_hz=row[1],
                max_length_in_s=audio_file_length_limit_in_s,
            ),
        )

        def reduce(series):
            merged_stats = None
            for audio_stats in series:
                if merged_stats is None:
                    merged_stats = audio_stats.copy()
                else:
                    AudioFeatureMixin._merge_stats(merged_stats, audio_stats)
            return merged_stats

        merged_stats = df_engine.reduce_objects(audio_stats, reduce)
        merged_stats["mean"] = calculate_mean(merged_stats["sum"], merged_stats["count"])
        merged_stats["var"] = calculate_var(merged_stats["sum"], merged_stats["sum2"], merged_stats["count"])
        merged_stats["std"] = np.sqrt(merged_stats["var"] / float(merged_stats["count"]))
        print_statistics = (
            "{} audio files loaded.\n"
            "Statistics of audio file lengths:\n"
            "- mean: {:.4f}\n"
            "- std: {:.4f}\n"
            "- max: {:.4f}\n"
            "- min: {:.4f}\n"
            "- cropped audio_files: {}\n"
            "Max length was given as {}s"
        ).format(
            merged_stats["count"],
            merged_stats["mean"],
            merged_stats["std"],
            merged_stats["max"],
            merged_stats["min"],
            merged_stats["cropped"],
            audio_file_length_limit_in_s,
        )
        logger.debug(print_statistics)
        return processed_audio

    @staticmethod
    def _transform_to_feature(
        audio: torch.Tensor,
        sampling_rate_in_hz: int,
        audio_feature_dict: Dict[str, Union[float, int, str]],
        feature_dim: int,
        max_length: int,
        padding_value: float,
        normalization_type: Optional[str] = None,
        type_key: str = TYPE,
    ):
        feature_type: str = str(audio_feature_dict[type_key])
        if feature_type == "raw":
            audio_feature = torch.unsqueeze(audio[0], dim=-1)
        elif feature_type in ["stft", "stft_phase", "group_delay", "fbank"]:
            audio_feature = AudioFeatureMixin._get_2D_feature(
                audio, feature_type, audio_feature_dict, sampling_rate_in_hz
            )
            audio_feature = torch.transpose(audio_feature, 0, 1)
        else:
            raise ValueError(f"{feature_type} is not recognized.")

        # Outer conditional is type refinement from Union[str, None] to str
        if normalization_type is not None:
            if normalization_type == "per_file":
                mean = torch.mean(audio_feature, dim=0)
                std = torch.std(audio_feature, dim=0)
                audio_feature = torch.divide((audio_feature - mean), std + 1.0e-10)
            elif normalization_type == "global":
                raise ValueError("not implemented yet")

        feature_length = audio_feature.shape[0]
        broadcast_feature_length = min(feature_length, max_length)
        audio_feature_padded = torch.full(
            (max_length, feature_dim), padding_value, dtype=torch.float32, device=audio_feature.device
        )
        audio_feature_padded[:broadcast_feature_length, :] = audio_feature[:max_length, :]

        return audio_feature_padded

    @staticmethod
    def _get_stats(audio, sampling_rate_in_hz, max_length_in_s):
        audio_length_in_s = audio.shape[-1] / float(sampling_rate_in_hz)
        return {
            "count": 1,
            "sum": audio_length_in_s,
            "sum2": audio_length_in_s * audio_length_in_s,
            "min": audio_length_in_s,
            "max": audio_length_in_s,
            "cropped": 1 if audio_length_in_s > max_length_in_s else 0,
        }

    @staticmethod
    def _merge_stats(merged_stats, audio_stats):
        merged_stats["count"] += audio_stats["count"]
        merged_stats["sum"] += audio_stats["sum"]
        merged_stats["sum2"] += audio_stats["sum2"]
        merged_stats["min"] = min(merged_stats["min"], audio_stats["min"])
        merged_stats["max"] = max(merged_stats["max"], audio_stats["max"])
        merged_stats["cropped"] += audio_stats["cropped"]

    @staticmethod
    def _get_2D_feature(
        audio: torch.Tensor,
        feature_type: str,
        audio_feature_dict: Dict[str, Union[float, int, str]],
        sampling_rate_in_hz: int,
    ) -> torch.Tensor:
        window_length_in_s = audio_feature_dict["window_length_in_s"]
        window_shift_in_s = audio_feature_dict["window_shift_in_s"]
        assert torch.jit.isinstance(window_length_in_s, float)
        assert torch.jit.isinstance(window_shift_in_s, float)

        window_length_in_samp = get_length_in_samp(window_length_in_s, sampling_rate_in_hz)

        if "num_fft_points" in audio_feature_dict:
            num_fft_points = audio_feature_dict["num_fft_points"]
            assert torch.jit.isinstance(num_fft_points, int)

            if num_fft_points < window_length_in_samp:
                raise ValueError(
                    "num_fft_points: {} < window length in "
                    "samples: {} (corresponds to window length"
                    " in s: {}".format(num_fft_points, window_length_in_s, window_length_in_samp)
                )
        else:
            num_fft_points = window_length_in_samp

        if "window_type" in audio_feature_dict:
            window_type = audio_feature_dict["window_type"]
            assert torch.jit.isinstance(window_type, str)
        else:
            window_type = "hamming"

        if feature_type == "stft_phase":
            return get_phase_stft_magnitude(
                audio, sampling_rate_in_hz, window_length_in_s, window_shift_in_s, num_fft_points, window_type
            )
        elif feature_type == "stft":
            return get_stft_magnitude(
                audio, sampling_rate_in_hz, window_length_in_s, window_shift_in_s, num_fft_points, window_type
            )
        elif feature_type == "group_delay":
            return get_group_delay(
                audio, sampling_rate_in_hz, window_length_in_s, window_shift_in_s, num_fft_points, window_type
            )
        elif feature_type == "fbank":
            num_filter_bands = audio_feature_dict["num_filter_bands"]
            assert torch.jit.isinstance(num_filter_bands, int)

            return get_fbank(
                audio,
                sampling_rate_in_hz,
                window_length_in_s,
                window_shift_in_s,
                num_fft_points,
                window_type,
                num_filter_bands,
            )
        else:
            raise ValueError(f'feature_type "{feature_type}" is not recognized.')

    @staticmethod
    def add_feature_data(
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input,
    ):
        set_default_value(feature_config["preprocessing"], "in_memory", preprocessing_parameters["in_memory"])

        name = feature_config[NAME]
        column = input_df[feature_config[COLUMN]]

        num_audio_files = len(column)
        if num_audio_files == 0:
            raise ValueError("There are no audio files in the dataset provided.")

        first_audio_entry = next(iter(column))
        logger.debug(f"Detected audio feature type is {type(first_audio_entry)}")

        if not isinstance(first_audio_entry, str) and not isinstance(first_audio_entry, torch.Tensor):
            raise ValueError(
                "Invalid audio feature data type.  Detected type is {}, "
                "expected either string for local/remote file path or Torch Tensor.".format(type(first_audio_entry))
            )

        src_path = None
        if SRC in metadata:
            if isinstance(first_audio_entry, str) and not has_remote_protocol(first_audio_entry):
                src_path = os.path.dirname(os.path.abspath(metadata.get(SRC)))
        abs_path_column = backend.df_engine.map_objects(  # This gets the CSV file path
            column, lambda row: get_abs_path(src_path, row) if isinstance(row, str) else row
        )

        num_audio_utterances = len(input_df[feature_config[COLUMN]])
        padding_value = preprocessing_parameters["padding_value"]
        normalization_type = preprocessing_parameters["norm"]

        feature_dim = metadata[name]["feature_dim"]
        max_length = metadata[name]["max_length"]
        audio_feature_dict = {
            key: value
            for key, value in preprocessing_parameters.items()
            if key in AUDIO_FEATURE_KEYS and value is not None
        }
        audio_file_length_limit_in_s = preprocessing_parameters["audio_file_length_limit_in_s"]

        if num_audio_utterances == 0:
            raise ValueError("There are no audio files in the dataset provided.")

        if feature_config[PREPROCESSING]["in_memory"]:
            audio_features = AudioFeatureMixin._process_in_memory(
                abs_path_column,
                audio_feature_dict,
                feature_dim,
                max_length,
                padding_value,
                normalization_type,
                audio_file_length_limit_in_s,
                backend,
            )
            proc_df[feature_config[PROC_COLUMN]] = audio_features

        return proc_df

    @staticmethod
    def _get_max_length_feature(
        preprocessing_parameters: PreprocessingConfigDict, sampling_rate_in_hz, audio_length_limit_in_s
    ):
        feature_type = preprocessing_parameters[TYPE]
        audio_length_limit_in_samp = audio_length_limit_in_s * sampling_rate_in_hz

        if not audio_length_limit_in_samp.is_integer():
            raise ValueError(
                "Audio_file_length_limit has to be chosen "
                "so that {} (in s) * {} (sampling rate in Hz) "
                "is an integer.".format(audio_length_limit_in_s, sampling_rate_in_hz)
            )
        audio_length_limit_in_samp = int(audio_length_limit_in_samp)

        if feature_type == "raw":
            return audio_length_limit_in_samp
        elif feature_type in ["stft", "stft_phase", "group_delay", "fbank"]:
            window_length_in_s = preprocessing_parameters["window_length_in_s"]
            window_shift_in_s = preprocessing_parameters["window_shift_in_s"]
            return get_max_length_stft_based(
                audio_length_limit_in_samp, window_length_in_s, window_shift_in_s, sampling_rate_in_hz
            )
        else:
            raise ValueError(f"{feature_type} is not recognized.")


class AudioInputFeature(AudioFeatureMixin, SequenceInputFeature):
    def __init__(self, input_feature_config: AudioInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, encoder_obj=encoder_obj, **kwargs)

        if not getattr(self.encoder_obj.config, "embedding_size", None):
            raise ValueError("embedding_size has to be defined - " 'check "update_config_with_metadata()"')
        if not getattr(self.encoder_obj.config, "max_sequence_length", None):
            raise ValueError("max_sequence_length has to be defined - " 'check "update_config_with_metadata()"')

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype == torch.float32
        assert len(inputs.shape) == 3, f"expected 3D shape, found: {inputs.shape}"

        encoder_output = self.encoder_obj(inputs, mask=mask)

        return encoder_output

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.encoder_obj.config.max_sequence_length, self.encoder_obj.config.embedding_size])

    @property
    def input_dtype(self):
        return torch.float32

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        feature_config.encoder.max_sequence_length = feature_metadata["max_length"]
        feature_config.encoder.embedding_size = feature_metadata["feature_dim"]
        feature_config.encoder.should_embed = False

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _AudioPreprocessing(metadata)

    @staticmethod
    def get_schema_cls():
        return AudioInputFeatureConfig
