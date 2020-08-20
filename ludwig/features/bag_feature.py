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
from collections import Counter

import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.encoders.bag_encoders import BagEmbedWeightedEncoder
from ludwig.features.base_feature import InputFeature
from ludwig.features.feature_utils import set_str_to_idx
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.strings_utils import create_vocabulary, UNKNOWN_SYMBOL

logger = logging.getLogger(__name__)


class BagFeatureMixin(object):
    type = BAG

    preprocessing_defaults = {
        'tokenizer': 'space',
        'most_common': 10000,
        'lowercase': False,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': UNKNOWN_SYMBOL
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        idx2str, str2idx, str2freq, max_size = create_vocabulary(
            column,
            preprocessing_parameters['tokenizer'],
            num_most_frequent=preprocessing_parameters['most_common'],
            lowercase=preprocessing_parameters['lowercase']
        )
        return {
            'idx2str': idx2str,
            'str2idx': str2idx,
            'str2freq': str2freq,
            'vocab_size': len(str2idx),
            'max_set_size': max_size
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters):
        bag_matrix = np.zeros(
            (len(column),
             len(metadata['str2idx'])),
            dtype=np.float32
        )

        for i, set_str in enumerate(column):
            col_counter = Counter(set_str_to_idx(
                set_str,
                metadata['str2idx'],
                preprocessing_parameters['tokenizer'])
            )
            bag_matrix[i, list(col_counter.keys())] = list(
                col_counter.values())

        return bag_matrix

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters=None
    ):
        data[feature['name']] = BagFeatureMixin.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']],
            preprocessing_parameters
        )


class BagInputFeature(BagFeatureMixin, InputFeature):
    encoder = 'embed'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        # assert inputs.dtype == tf.bool # this fails

        encoder_output = self.encoder_obj(inputs, training=training, mask=mask)

        return {'encoder_output': encoder_output}

    def get_input_dtype(self):
        return tf.float32

    def get_input_shape(self):
        return len(self.vocab),

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        input_feature['vocab'] = feature_metadata['idx2str']

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = {
        'embed': BagEmbedWeightedEncoder,
        None: BagEmbedWeightedEncoder
    }
