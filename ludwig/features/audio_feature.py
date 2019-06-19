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

import h5py
import numpy as np
import tensorflow as tf
import soundfile
import ipdb

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.utils.image_utils import get_abs_path
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
        'normalization': None,
        'audio_feature': {
            'type': 'raw'
            'dim': 1
        }
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        max_length = 0
        for audio_file_path in column:
            audio, _ = soundfile.read(audio_file_path)
            max_length = max(max_length, audio.shape[-1])

        return { 'max_audio_length': max_length }

    @staticmethod
    def _read_audio(filepath):
        """
        :param filepath: path to the audio
        """
        audio, sampling_rate_in_hz = soundfile.read(filepath)
        return audio, sampling_rate_in_hz

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        ipdb.set_trace()
        set_default_value(
            feature['preprocessing'],
            'in_memory',
            preprocessing_parameters['in_memory']
        )

        feature_name = feature['name']
        max_audio_length = metadata[feature_name]['max_audio_length']

        assert 'audio_feature' in preprocessing_parameters
        audio_feature = preprocessing_parameters['audio_feature']
        assert 'type' in audio_feature
        feature_type = audio_feature['type']
        assert 'dim' in audio_feature
        feature_dim = audio_feature['dim']

        csv_path = None
        if hasattr(dataset_df, 'csv'):
            csv_path = os.path.dirname(os.path.abspath(dataset_df.csv))

        num_audio_utterances = len(dataset_df)

        if num_audio_utterances == 0:
            raise ValueError('There are no audio files in the dataset provided.')
        if (csv_path is None and
                not os.path.isabs(dataset_df[feature['name']][0])):
            raise ValueError(
                'Audio file paths must be absolute'
            )

        if feature['preprocessing']['in_memory']:
            data[feature['name']] = np.empty(
                (num_audio_utterances, feature_dim, length),
                dtype=np.float32
            )
            for i in range(len(dataset_df)):
                filepath = get_abs_path(
                    csv_path,
                    dataset_df[feature['name']][i]
                )

                audio = AudioBaseFeature._read_audio(filepath)
                data[feature['name']][i, :, :] = audio


class AudioInputFeature(AudioBaseFeature, SequenceInputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = AUDIO

    def _get_input_placeholder(self):
        return tf.placeholder(
            tf.float32, shape=[None, self.feature_dim, self.length],
            name='{}_placeholder'.format(self.name)
        )

    def build_input(
            self, 
            regularizer,
            dropout_rate,
            is_training=False,
            **kwargs
    ):
        placeholder = self.__get_input_placeholder()
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
        input_feature['length'] = self.length
        input_feature['embedding_size'] = self.feature_dim
        input_feature['should_embed'] = False

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)
        set_default_value(input_feature, 'preprocessing', {})
