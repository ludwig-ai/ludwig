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

import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.models.modules.h3_encoders import H3WeightedSum, H3RNN, H3Embed
from ludwig.utils.h3_util import h3_to_components
from ludwig.utils.misc import set_default_value, get_from_registry

logger = logging.getLogger(__name__)

MAX_H3_RESOLUTION = 15
H3_VECTOR_LENGTH = MAX_H3_RESOLUTION + 4
H3_PADDING_VALUE = 7


class H3BaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = H3

    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 576495936675512319
        # mode 1 edge 0 resolution 0 base_cell 0
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {}

    @staticmethod
    def h3_to_list(h3_int):
        components = h3_to_components(h3_int)
        header = [
            components['mode'],
            components['edge'],
            components['resolution'],
            components['base_cell']
        ]
        cells_padding = [H3_PADDING_VALUE] * (
                MAX_H3_RESOLUTION - len(components['cells'])
        )
        return header + components['cells'] + cells_padding

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters=None
    ):
        data[feature['name']] = np.array(
            [H3BaseFeature.h3_to_list(row)
             for row in dataset_df[feature['name']]], dtype=np.uint8
        )


class H3InputFeature(H3BaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.encoder = 'embed'

        encoder_parameters = self.overwrite_defaults(feature)

        self.encoder_obj = self.get_h3_encoder(encoder_parameters)

    def get_h3_encoder(self, encoder_parameters):
        return get_from_registry(
            self.encoder, h3_encoder_registry)(
            **encoder_parameters
        )

    def _get_input_placeholder(self):
        # None dimension is for dealing with variable batch size
        return tf.compat.v1.placeholder(
            tf.int32,
            shape=[None, H3_VECTOR_LENGTH],
            name=self.name
        )

    def build_input(
            self,
            regularizer,
            dropout_rate,
            is_training=False,
            **kwargs
    ):
        placeholder = self._get_input_placeholder()
        logger.debug('placeholder: {0}'.format(placeholder))

        feature_representation, feature_representation_size = self.encoder_obj(
            placeholder,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )
        logging.debug('  feature_representation: {0}'.format(
            feature_representation))

        feature_representation = {
            'name': self.name,
            'type': self.type,
            'representation': feature_representation,
            'size': feature_representation_size,
            'placeholder': placeholder
        }
        return feature_representation

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)


h3_encoder_registry = {
    'embed': H3Embed,
    'weighted_sum': H3WeightedSum,
    'rnn': H3RNN
}
