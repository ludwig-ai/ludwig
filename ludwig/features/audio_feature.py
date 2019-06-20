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

import numpy as np
import tensorflow as tf
import soundfile

from ludwig.constants import AUDIO, BACKFILL
from ludwig.features.base_feature import BaseFeature
from ludwig.features.sequence_feature import SequenceInputFeature

from ludwig.utils.image_utils import get_abs_path
from ludwig.utils.audio_utils import get_num_output_padded_to_fit_input
from ludwig.utils.audio_utils import get_length_in_samp
from ludwig.utils.audio_utils import get_group_delay
from ludwig.utils.audio_utils import get_phase_stft_magnitude
from ludwig.utils.audio_utils import get_stft_magnitude
from ludwig.utils.audio_utils import get_non_symmetric_length
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import set_default_value


logger = logging.getLogger(__name__)


class AudioBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = AUDIO

    preprocessing_defaults = {
        'missing_value_strategy': BACKFILL,
        'in_memory': True,
        'padding_value': None,
        'normalization': None,
        'audio_feature': {
           'type': 'raw',
        }
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        audio_feature_dict = preprocessing_parameters['audio_feature']
        first_audio_file_path = column[0]
        _, sampling_rate_in_hz = soundfile.read(first_audio_file_path)

        feature_dim = AudioBaseFeature._get_feature_dim(audio_feature_dict, sampling_rate_in_hz)
        max_utterance_length = AudioBaseFeature._get_max_utterance_length(column, audio_feature_dict, sampling_rate_in_hz)

        return { 
                'feature_dim': feature_dim,
                'max_utterance_length': max_utterance_length,
                'sampling_rate_in_hz': sampling_rate_in_hz 
               }

    @staticmethod
    def _get_feature_dim(audio_feature_dict, sampling_rate_in_hz):
        feature_type = audio_feature_dict['type']

        if(feature_type == 'raw'):
            feature_dim = 1
        elif(feature_type == 'stft_phase'):
            feature_dim_symmetric = get_length_in_samp(audio_feature_dict['window_length_in_s'], sampling_rate_in_hz)
            feature_dim = 2 * get_non_symmetric_length(feature_dim_symmetric)
        elif(feature_type in ['stft', 'group_delay']):
            feature_dim_symmetric = get_length_in_samp(audio_feature_dict['window_length_in_s'], sampling_rate_in_hz)
            feature_dim = get_non_symmetric_length(feature_dim_symmetric)
        else:
            raise ValueError('{} is not recognized.'.format(feature_type))

        return feature_dim

    @staticmethod
    def _get_max_utterance_length(column, audio_feature_dict, sampling_rate_in_hz_check):
        feature_type = audio_feature_dict['type']
        max_length = 0
        for audio_file_path in column:
            audio, sampling_rate_in_hz = soundfile.read(audio_file_path)

            if(sampling_rate_in_hz != sampling_rate_in_hz_check):
                raise ValueError('Sampling rate of {} is {}, but should be {}'.format(audio_file_path, sampling_rate_in_hz, sampling_rate_in_hz_check))

            audio_length = audio.shape[-1]

            if(feature_type == 'raw'):
                feature_length = audio_length
            elif(feature_type in ['stft', 'group_delay', 'stft_phase']):
                window_length_in_samp = get_length_in_samp(audio_feature_dict['window_length_in_s'], sampling_rate_in_hz)
                window_shift_in_samp = get_length_in_samp(audio_feature_dict['window_shift_in_s'], sampling_rate_in_hz)
                feature_length = get_num_output_padded_to_fit_input(audio_length, window_length_in_samp, window_shift_in_samp)
            else:
                raise ValueError('{} is not recognized.'.format(feature_type))

            max_length = max(max_length, feature_length)
        return max_length

    @staticmethod
    def _read_audio_and_transform_to_feature(filepath, audio_feature_dict, feature_dim, max_utterance_length, padding_value):
        """
        :param filepath: path to the audio
        :param audio_feature_dict: dictionary describing audio feature see default
        :param feature_dim: dimension of each feature frame
        :param max_utterance_length: max length of all utterances
        """

        feature_type = audio_feature_dict['type']
        audio, sampling_rate_in_hz = soundfile.read(filepath)
        audio_feature_padded = np.full((feature_dim, max_utterance_length), padding_value)

        if(feature_type == 'raw'):
            audio_feature = np.expand_dims(audio, axis=0)
        elif(feature_type in ['stft', 'stft_phase', 'group_delay']):
            audio_feature = AudioBaseFeature._get_2D_feature(audio, feature_type, audio_feature_dict, sampling_rate_in_hz)
        else:
            raise ValueError('{} is not recognized.'.format(feature_type))

        feature_length = audio_feature.shape[-1]
        audio_feature_padded[:,:feature_length] = audio_feature

        return audio_feature_padded

    @staticmethod
    def _get_2D_feature(audio, feature_type, audio_feature_dict, sampling_rate_in_hz):
        window_length_in_s = audio_feature_dict['window_length_in_s']
        window_shift_in_s = audio_feature_dict['window_shift_in_s']
        num_fft_points = audio_feature_dict['num_fft_points'] if 'num_fft_points' in audio_feature_dict else get_length_in_samp(window_length_in_s, sampling_rate_in_hz)
        window_type = audio_feature_dict['window_type'] if 'window_type' in audio_feature_dict else 'hamming'

        if(feature_type == 'stft_phase'):
            return get_phase_stft_magnitude(audio, sampling_rate_in_hz, window_length_in_s, window_shift_in_s, num_fft_points, window_type)
        if(feature_type == 'stft'):
            return get_stft_magnitude(audio, sampling_rate_in_hz, window_length_in_s, window_shift_in_s, num_fft_points, window_type)
        if(feature_type == 'group_delay'):
            return get_group_delay(audio, sampling_rate_in_hz, window_length_in_s, window_shift_in_s, num_fft_points, window_type)

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

        if(not 'audio_feature' in preprocessing_parameters):
            raise ValueError('audio_feature dictionary has to be present in preprocessing for audio.')
        if(not 'type' in preprocessing_parameters['audio_feature']):
            raise ValueError('type key has to be present in audio_feature dictionary for audio.')

        csv_path = None
        if hasattr(dataset_df, 'csv'):
            csv_path = os.path.dirname(os.path.abspath(dataset_df.csv))
        if (csv_path is None and
                not os.path.isabs(dataset_df[feature['name']][0])):
            raise ValueError(
                'Audio file paths must be absolute'
            )

        num_audio_utterances = len(dataset_df)
        padding_value = preprocessing_parameters['padding_value']
        feature_name = feature['name']
        max_utterance_length = metadata[feature_name]['max_utterance_length']
        feature_dim = metadata[feature_name]['feature_dim']
        audio_feature_dict = preprocessing_parameters['audio_feature']

        if num_audio_utterances == 0:
            raise ValueError('There are no audio files in the dataset provided.')

        if feature['preprocessing']['in_memory']:
            data[feature['name']] = np.empty(
                (num_audio_utterances, feature_dim, max_utterance_length),
                dtype=np.float32
            )
            for i in range(len(dataset_df)):
                filepath = get_abs_path(
                    csv_path,
                    dataset_df[feature['name']][i]
                )
                audio_feature = AudioBaseFeature._read_audio_and_transform_to_feature(filepath, audio_feature_dict, feature_dim, max_utterance_length, padding_value)
                # TODO: add optional normalization step here
                data[feature['name']][i, :, :] = audio_feature


class AudioInputFeature(AudioBaseFeature, SequenceInputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = AUDIO
        self.length = None
        self.embedding_size = None
        self.overwrite_defaults(feature)
        if(not self.embedding_size):
            raise ValueError('embedding_size have to be defined - check "update_model_definition_with_metadata()"')
        if(not self.length):
            raise ValueError('length have to be defined - check "update_model_definition_with_metadata()"')

    def _get_input_placeholder(self):
        return tf.placeholder(
            tf.float32, shape=[None, self.embedding_size, self.length],
            name='{}_placeholder'.format(self.name)
        )

    def build_input(
            self, 
            regularizer,
            dropout_rate,
            is_training=False,
            **kwargs
    ):
        placeholder = self._get_input_placeholder()
        logger.debug('  placeholder: {0}'.format(placeholder))

        return self.build_sequence_input(
                placeholder, 
                self.encoder_obj,
                regularizer,
                dropout_rate,
                is_training
        )

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        input_feature['length'] = feature_metadata['max_utterance_length']
        input_feature['embedding_size'] = feature_metadata['feature_dim']
        input_feature['should_embed'] = False

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)
        set_default_value(input_feature, 'preprocessing', {})
