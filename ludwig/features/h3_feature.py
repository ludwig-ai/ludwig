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
from ludwig.encoders.h3_encoders import ENCODER_REGISTRY
from ludwig.features.base_feature import InputFeature
from ludwig.utils.h3_util import h3_to_components
from ludwig.utils.misc_utils import set_default_value

logger = logging.getLogger(__name__)

MAX_H3_RESOLUTION = 15
H3_VECTOR_LENGTH = MAX_H3_RESOLUTION + 4
H3_PADDING_VALUE = 7


class H3FeatureMixin(object):
    type = H3
    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 576495936675512319
        # mode 1 edge 0 resolution 0 base_cell 0
    }

    @staticmethod
    def cast_column(feature, dataset_df, backend):
        # todo: add cast to int64
        return dataset_df

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
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
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend
    ):
        column = input_df[feature[COLUMN]]
        if column.dtype == object:
            column = column.map(int)
        column = column.map(H3FeatureMixin.h3_to_list)

        proc_df[feature[PROC_COLUMN]] = backend.df_engine.map_objects(
            column,
            lambda x: np.array(x, dtype=np.uint8)
        )
        return proc_df


class H3InputFeature(H3FeatureMixin, InputFeature):
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
        assert inputs.dtype == tf.uint8
        assert len(inputs.shape) == 2

        inputs_encoded = self.encoder_obj(
            inputs, training=training, mask=mask
        )

        return inputs_encoded

    @classmethod
    def get_input_dtype(cls):
        return tf.uint8

    def get_input_shape(self):
        return H3_VECTOR_LENGTH,

    @staticmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = ENCODER_REGISTRY
