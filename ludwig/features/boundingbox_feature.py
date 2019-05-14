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
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.fully_connected_modules import fc_layer
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.measure_modules import \
    absolute_error as get_absolute_error
from ludwig.models.modules.measure_modules import error as get_error
from ludwig.models.modules.measure_modules import r2 as get_r2
from ludwig.models.modules.measure_modules import \
    squared_error as get_squared_error
from ludwig.utils.misc import set_default_value



class BoundingBoxBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = NUMERICAL

    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 0
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {}

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters,
    ):
        data[feature['name']] = dataset_df[feature['name']].astype(
            np.float32).as_matrix()



class BoundingBoxOutputFeature(BoundingBoxBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.loss = {'type': MEAN_SQUARED_ERROR}
        self.clip = None
        self.initializer = None
        self.regularize = True
        self.box_coordinates = [None, None, None, None]

        _ = self.overwrite_defaults(feature)

    def _get_output_placeholder(self):
        return tf.placeholder(
            tf.int64,
            [None, len(self.box_coordinates)],  # None is for dealing with variable batch size
            name='{}_placeholder'.format(self.name)
        )

    def _get_predictions(
            self,
            hidden,
            hidden_size,
            regularizer=None
    ):
        if not self.regularize:
            regularizer = None

        with tf.variable_scope('predictions_{}'.format(self.name)):
            initializer_obj = get_initializer(self.initializer)
            weights = tf.get_variable(
                'weights',
                initializer=initializer_obj([hidden_size, 1]),
                regularizer=regularizer
            )
            logging.debug('  regression_weights: {0}'.format(weights))

            biases = tf.get_variable('biases', [1])
            logging.debug('  regression_biases: {0}'.format(biases))

            predictions = tf.reshape(
                tf.matmul(hidden, weights) + biases,
                [-1]
            )
            logging.debug('  predictions: {0}'.format(predictions))

            if self.clip is not None:
                if isinstance(self.clip, (list, tuple)) and len(self.clip) == 2:
                    predictions = tf.clip_by_value(
                        predictions,
                        self.clip[0],
                        self.clip[1]
                    )
                    logging.debug(
                        '  clipped_predictions: {0}'.format(predictions)
                    )
                else:
                    raise ValueError(
                        'The clip parameter of {} is {}. '
                        'It must be a list or a tuple of length 2.'.format(
                            self.name,
                            self.clip
                        )
                    )

        return predictions

    def _get_loss(self, targets, predictions):
    	pass

    def _get_measures(self, targets, predictions):
    	pass

    def build_output(
            self,
            hidden,
            hidden_size,
            regularizer=None,
            **kwargs
    ):
    	pass

    @staticmethod
    def update_model_definition_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def calculate_overall_stats(
            test_stats,
            output_feature,
            dataset,
            train_set_metadata
    ):
        pass

    @staticmethod
    def postprocess_results(
            output_feature,
            result,
            metadata,
            experiment_dir_name,
            skip_save_unprocessed_output=False
    ):
    	pass

    @staticmethod
    def populate_defaults(output_feature):
    	pass

