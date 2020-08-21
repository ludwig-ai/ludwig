#! /usr/bin/env python
# coding=utf-8
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
import sys

import numpy as np
import tensorflow as tf

from ludwig.constants import AUDIO, BACKFILL, TIED, TYPE
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.utils.audio_utils import calculate_incr_mean
from ludwig.utils.audio_utils import calculate_incr_var
from ludwig.utils.audio_utils import get_fbank
from ludwig.utils.audio_utils import get_group_delay
from ludwig.utils.audio_utils import get_length_in_samp
from ludwig.utils.audio_utils import get_max_length_stft_based
from ludwig.utils.audio_utils import get_non_symmetric_length
from ludwig.utils.audio_utils import get_phase_stft_magnitude
from ludwig.utils.audio_utils import get_stft_magnitude
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.misc_utils import set_default_values

logger = logging.getLogger(__name__)


class AudioFeatureMixin(object):
    type = AUDIO

    preprocessing_defaults = {
        'audio_file_length_limit_in_s': 7.5,
        'missing_value_strategy': BACKFILL,
        'in_memory': True,
        'padding_value': 0,
        'norm': None,
        'audio_feature': {
            'type': 'raw',
        }
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        try:
            import soundfile
        except ImportError:
            logger.error(
                ' soundfile is not installed. '
                'In order to install all audio feature dependencies run '
                'pip install ludwig[audio]'
            )
            sys.exit(-1)

        audio_feature_dict = preprocessing_parameters['audio_feature']
        first_audio_file_path = column[0]
        _, sampling_rate_in_hz = soundfile.read(first_audio_file_path)

        feature_dim = AudioFeatureMixin._get_feature_dim(audio_feature_dict,
                                                         sampling_rate_in_hz)
        audio_file_length_limit_in_s = preprocessing_parameters[
            'audio_file_length_limit_in_s']
        max_length = AudioFeatureMixin._get_max_length_feature(
            audio_feature_dict, sampling_rate_in_hz,
            audio_file_length_limit_in_s)
        return {
            'feature_dim': feature_dim,
            'sampling_rate_in_hz': sampling_rate_in_hz,
            'max_length': max_length
        }

    @staticmethod
    def _get_feature_dim(audio_feature_dict, sampling_rate_in_hz):
        feature_type = audio_feature_dict[TYPE]

        if feature_type == 'raw':
            feature_dim = 1
        elif feature_type == 'stft_phase':
            feature_dim_symmetric = get_length_in_samp(
                audio_feature_dict['window_length_in_s'], sampling_rate_in_hz)
            feature_dim = 2 * get_non_symmetric_length(feature_dim_symmetric)
        elif feature_type in ['stft', 'group_delay']:
            feature_dim_symmetric = get_length_in_samp(
                audio_feature_dict['window_length_in_s'], sampling_rate_in_hz)
            feature_dim = get_non_symmetric_length(feature_dim_symmetric)
        elif feature_type == 'fbank':
            feature_dim = audio_feature_dict['num_filter_bands']
        else:
            raise ValueError('{} is not recognized.'.format(feature_type))

        return feature_dim

    @staticmethod
    def _read_audio_and_transform_to_feature(filepath, audio_feature_dict,
                                             feature_dim, max_length,
                                             padding_value, normalization_type,
                                             audio_stats):
        """
        :param filepath: path to the audio
        :param audio_feature_dict: dictionary describing audio feature see default
        :param feature_dim: dimension of each feature frame
        :param max_length: max audio length defined by user in samples
        """
        try:
            import soundfile
        except ImportError:
            logger.error(
                ' soundfile is not installed. '
                'In order to install all audio feature dependencies run '
                'pip install ludwig[audio]'
            )
            sys.exit(-1)

        feature_type = audio_feature_dict[TYPE]
        audio, sampling_rate_in_hz = soundfile.read(filepath)
        AudioFeatureMixin._update(audio_stats, audio, sampling_rate_in_hz)

        if feature_type == 'raw':
            audio_feature = np.expand_dims(audio, axis=-1)
        elif feature_type in ['stft', 'stft_phase', 'group_delay', 'fbank']:
            audio_feature = np.transpose(
                AudioFeatureMixin._get_2D_feature(audio, feature_type,
                                                  audio_feature_dict,
                                                  sampling_rate_in_hz))
        else:
            raise ValueError('{} is not recognized.'.format(feature_type))

        if normalization_type == 'per_file':
            mean = np.mean(audio_feature, axis=0)
            std = np.std(audio_feature, axis=0)
            audio_feature = np.divide((audio_feature - mean),
                                      std + 1.0e-10)
        elif normalization_type == 'global':
            raise ValueError('not implemented yet')

        feature_length = audio_feature.shape[0]
        broadcast_feature_length = min(feature_length, max_length)
        audio_feature_padded = np.full((max_length, feature_dim),
                                       padding_value,
                                       dtype=np.float32)
        audio_feature_padded[:broadcast_feature_length, :] = audio_feature[
                                                             :max_length, :]

        return audio_feature_padded

    @staticmethod
    def _update(audio_stats, audio, sampling_rate_in_hz):
        audio_length_in_s = audio.shape[-1] / float(sampling_rate_in_hz)
        audio_stats['count'] += 1
        mean = ((audio_stats['count'] - 1) * audio_stats[
            'mean'] + audio_length_in_s) / float(audio_stats['count'])
        mean = calculate_incr_mean(audio_stats['count'], audio_stats['mean'],
                                   audio_length_in_s)
        var = calculate_incr_var(audio_stats['var'], audio_stats['mean'], mean,
                                 audio_length_in_s)
        audio_stats['mean'] = mean
        audio_stats['var'] = var
        audio_stats['max'] = max(audio_stats['max'], audio_length_in_s)
        audio_stats['min'] = min(audio_stats['min'], audio_length_in_s)
        if audio_length_in_s > audio_stats['max_length_in_s']:
            audio_stats['cropped'] += 1

    @staticmethod
    def _get_2D_feature(audio, feature_type, audio_feature_dict,
                        sampling_rate_in_hz):
        window_length_in_s = audio_feature_dict['window_length_in_s']
        window_shift_in_s = audio_feature_dict['window_shift_in_s']
        window_length_in_samp = get_length_in_samp(window_length_in_s,
                                                   sampling_rate_in_hz)

        if 'num_fft_points' in audio_feature_dict:
            num_fft_points = audio_feature_dict['num_fft_points']
            if num_fft_points < window_length_in_samp:
                raise ValueError(
                    'num_fft_points: {} < window length in '
                    'samples: {} (corresponds to window length'
                    ' in s: {}'.format(num_fft_points, window_length_in_s,
                                       window_length_in_samp))
        else:
            num_fft_points = window_length_in_samp

        if 'window_type' in audio_feature_dict:
            window_type = audio_feature_dict['window_type']
        else:
            window_type = 'hamming'

        if feature_type == 'stft_phase':
            return get_phase_stft_magnitude(audio, sampling_rate_in_hz,
                                            window_length_in_s,
                                            window_shift_in_s, num_fft_points,
                                            window_type)
        if feature_type == 'stft':
            return get_stft_magnitude(audio, sampling_rate_in_hz,
                                      window_length_in_s, window_shift_in_s,
                                      num_fft_points, window_type)
        if feature_type == 'group_delay':
            return get_group_delay(audio, sampling_rate_in_hz,
                                   window_length_in_s, window_shift_in_s,
                                   num_fft_points, window_type)
        if feature_type == 'fbank':
            num_filter_bands = audio_feature_dict['num_filter_bands']
            return get_fbank(audio, sampling_rate_in_hz,
                             window_length_in_s, window_shift_in_s,
                             num_fft_points, window_type, num_filter_bands)

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        set_default_value(
            feature['preprocessing'],
            'in_memory',
            preprocessing_parameters['in_memory']
        )

        if not 'audio_feature' in preprocessing_parameters:
            raise ValueError(
                'audio_feature dictionary has to be present in preprocessing '
                'for audio.')
        if not TYPE in preprocessing_parameters['audio_feature']:
            raise ValueError(
                'type has to be present in audio_feature dictionary '
                'for audio.')

        csv_path = None
        # this is not super nice, but works both and DFs and lists
        first_path = '.'
        for first_path in dataset_df[feature['name']]:
            break
        if hasattr(dataset_df, 'csv'):
            csv_path = os.path.dirname(os.path.abspath(dataset_df.csv))
        if csv_path is None and not os.path.isabs(first_path):
            raise ValueError(
                'Audio file paths must be absolute'
            )

        num_audio_utterances = len(dataset_df)
        padding_value = preprocessing_parameters['padding_value']
        normalization_type = preprocessing_parameters['norm']
        feature_name = feature['name']

        feature_dim = metadata[feature_name]['feature_dim']
        max_length = metadata[feature_name]['max_length']
        audio_feature_dict = preprocessing_parameters['audio_feature']
        audio_file_length_limit_in_s = preprocessing_parameters[
            'audio_file_length_limit_in_s']

        if num_audio_utterances == 0:
            raise ValueError(
                'There are no audio files in the dataset provided.')
        audio_stats = {
            'count': 0,
            'mean': 0,
            'var': 0,
            'std': 0,
            'max': 0,
            'min': float('inf'),
            'cropped': 0,
            'max_length_in_s': audio_file_length_limit_in_s
        }

        if feature['preprocessing']['in_memory']:
            data[feature['name']] = np.empty(
                (num_audio_utterances, max_length, feature_dim),
                dtype=np.float32
            )
            for i, path in enumerate(dataset_df[feature['name']]):
                filepath = get_abs_path(
                    csv_path,
                    path
                )
                audio_feature = AudioFeatureMixin._read_audio_and_transform_to_feature(
                    filepath, audio_feature_dict, feature_dim, max_length,
                    padding_value, normalization_type, audio_stats
                )

                data[feature['name']][i, :, :] = audio_feature

            audio_stats['std'] = np.sqrt(
                audio_stats['var'] / float(audio_stats['count']))
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
                audio_stats['count'], audio_stats['mean'],
                audio_stats['std'], audio_stats['max'],
                audio_stats['min'], audio_stats['cropped'],
                audio_stats['max_length_in_s'])
            logger.debug(print_statistics)

    @staticmethod
    def _get_max_length_feature(
            audio_feature_dict,
            sampling_rate_in_hz,
            audio_length_limit_in_s
    ):
        feature_type = audio_feature_dict[TYPE]
        audio_length_limit_in_samp = (
                audio_length_limit_in_s * sampling_rate_in_hz
        )

        if not audio_length_limit_in_samp.is_integer():
            raise ValueError(
                'Audio_file_length_limit has to be chosen '
                'so that {} (in s) * {} (sampling rate in Hz) '
                'is an integer.'.format(
                    audio_length_limit_in_s, sampling_rate_in_hz))
        audio_length_limit_in_samp = int(audio_length_limit_in_samp)

        if feature_type == 'raw':
            return audio_length_limit_in_samp
        elif feature_type in ['stft', 'stft_phase', 'group_delay', 'fbank']:
            window_length_in_s = audio_feature_dict['window_length_in_s']
            window_shift_in_s = audio_feature_dict['window_shift_in_s']
            return get_max_length_stft_based(audio_length_limit_in_samp,
                                             window_length_in_s,
                                             window_shift_in_s,
                                             sampling_rate_in_hz)
        else:
            raise ValueError('{} is not recognized.'.format(feature_type))


class AudioInputFeature(AudioFeatureMixin, SequenceInputFeature):
    length = None
    embedding_size = None
    encoder = 'embed'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature, encoder_obj=encoder_obj)
        if not self.embedding_size:
            raise ValueError(
                'embedding_size has to be defined - '
                'check "update_model_definition_with_metadata()"')
        if not self.length:
            raise ValueError(
                'length has to be defined - '
                'check "update_model_definition_with_metadata()"')

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.float32
        assert len(inputs.shape) == 3

        encoder_output = self.encoder_obj(
            inputs, training=training, mask=mask
        )

        return encoder_output

    def get_input_dtype(self):
        return tf.float32

    def get_input_shape(self):
        return self.length, self.embedding_size

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        input_feature['length'] = feature_metadata['max_length']
        input_feature['embedding_size'] = feature_metadata['feature_dim']
        input_feature['should_embed'] = False

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(
            input_feature,
            {
                TIED: None,
                'preprocessing': {}
            }
        )
