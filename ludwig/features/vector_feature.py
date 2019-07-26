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

import h5py
import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.models.modules.dense_encoders import FCStack
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import set_default_value

logger = logging.getLogger(__name__)


class VectorBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = VECTOR

    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {
            'preprocessing': preprocessing_parameters
        }

    @staticmethod
    def read_single_vector(row):
        return [float(x) for x in row.split()]

    @staticmethod
    def feature_data(column, metadata):
        vectors = column.map(VectorBaseFeature.read_single_vector)
        for v in vectors:
            if len(v) != metadata['vector_size']:
                raise ValueError(
                    'All the vectors need to be of the same size. Expected size:'
                    '{}. Actual Size: {}'.format(metadata['vector_size'], len(v))
                )
        return np.array(vectors)

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        num_vectors = len(dataset_df)
        if num_vectors == 0:
            raise ValueError("There are no vectors in the dataset provided")

        if 'vector_size' not in preprocessing_parameters:
            vector_size = len(VectorBaseFeature.read_single_vector(
                dataset_df[feature['name']][0]
            ))
        else:
            vector_size = preprocessing_parameters['vector_size']

        metadata[feature['name']]['preprocessing']['vector_size'] = vector_size

        data[feature['name']] = VectorBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']]
        )


class VectorInputFeature(VectorBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.vector_size = 0
        self.encoder = 'fc_stack'

        encoder_parameters = self.overwrite_defaults(feature)

        self.encoder_obj = self.get_vector_encoder(encoder_parameters)

    def get_vector_encoder(self, encoder_parameters):
        return get_from_registry(self.encoder, vector_encoder_registry)(
            **encoder_parameters
        )

    def _get_input_placeholder(self):
        # None dimension is for dealing with variable batch size
        return tf.placeholder(
            tf.float32,
            shape=[None, self.vector_size],
            name=self.name,
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

        feature_representation, feature_representation_size = self.encoder_obj(
            placeholder,
            regularizer,
            dropout_rate,
            is_training,
        )
        logger.debug(
            '  feature_representation: {0}'.format(feature_representation)
        )

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
        for key in ['vector_size']:
            input_feature[key] = feature_metadata['preprocessing'][key]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)
        set_default_value(input_feature, 'preprocessing', {})


vector_encoder_registry = {
    'fc_stack': FCStack
}

