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
from ludwig.features.base_feature import InputFeature
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


class NumericalBaseFeature(BaseFeature):
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


class NumericalInputFeature(NumericalBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.norm = None
        self.dropout = False

        _ = self.overwrite_defaults(feature)

    def _get_input_placeholder(self):
        return tf.placeholder(
            tf.float32,
            shape=[None],  # None is for dealing with variable batch size
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

        feature_representation = fc_layer(
            tf.expand_dims(tf.cast(placeholder, tf.float32), 1),
            1,
            1,
            activation=None,
            norm=self.norm,
            dropout=self.dropout,
            dropout_rate=dropout_rate,
            regularizer=regularizer,
            initializer='ones'
        )

        logging.debug('  feature_representation: {0}'.format(
            feature_representation))

        feature_representation = {'type': self.name,
                                  'representation': feature_representation,
                                  'size': 1,
                                  'placeholder': placeholder}

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


class NumericalOutputFeature(NumericalBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.loss = {'type': MEAN_SQUARED_ERROR}
        self.clip = None
        self.initializer = None
        self.regularize = True

        _ = self.overwrite_defaults(feature)

    def _get_output_placeholder(self):
        return tf.placeholder(
            tf.float32,
            [None],  # None is for dealing with variable batch size
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
        with tf.variable_scope('loss_{}'.format(self.name)):
            if self.loss['type'] == 'mean_squared_error':
                train_loss = tf.losses.mean_squared_error(
                    labels=targets,
                    predictions=predictions,
                    reduction=Reduction.NONE)
            elif self.loss['type'] == 'mean_absolute_error':
                train_loss = tf.losses.absolute_difference(
                    labels=targets,
                    predictions=predictions,
                    reduction=Reduction.NONE
                )
            else:
                train_mean_loss = None
                train_loss = None
                raise ValueError(
                    'Unsupported loss type {}'.format(self.loss['type'])
                )

            train_mean_loss = tf.reduce_mean(
                train_loss,
                name='train_mean_loss_{}'.format(self.name)
            )

        return train_mean_loss, train_loss

    def _get_measures(self, targets, predictions):

        with tf.variable_scope('measures_{}'.format(self.name)):
            error_val = get_error(
                targets,
                predictions,
                self.name
            )

            absolute_error_val = get_absolute_error(
                targets,
                predictions,
                self.name
            )

            squared_error_val = get_squared_error(
                targets,
                predictions,
                self.name
            )

            r2_val = get_r2(targets, predictions, self.name)

        return error_val, squared_error_val, absolute_error_val, r2_val

    def build_output(
            self,
            hidden,
            hidden_size,
            regularizer=None,
            **kwargs
    ):
        output_tensors = {}

        # ================ Placeholder ================
        targets = self._get_output_placeholder()
        output_tensors[self.name] = targets
        logging.debug('  targets_placeholder: {0}'.format(targets))

        # ================ Predictions ================
        predictions = self._get_predictions(
            hidden,
            hidden_size
        )

        output_tensors[PREDICTIONS + '_' + self.name] = predictions

        # ================ Measures ================
        error, squared_error, absolute_error, r2 = self._get_measures(
            targets,
            predictions
        )

        output_tensors[ERROR + '_' + self.name] = error
        output_tensors[SQUARED_ERROR + '_' + self.name] = squared_error
        output_tensors[ABSOLUTE_ERROR + '_' + self.name] = absolute_error
        output_tensors[R2 + '_' + self.name] = r2

        if 'sampled' not in self.loss['type']:
            tf.summary.scalar(
                'train_batch_mean_squared_error_{}'.format(self.name),
                tf.reduce_mean(squared_error)
            )
            tf.summary.scalar(
                'train_batch_mean_absolute_error_{}'.format(self.name),
                tf.reduce_mean(absolute_error)
            )
            tf.summary.scalar(
                'train_batch_mean_r2_{}'.format(self.name),
                tf.reduce_mean(r2)
            )

        # ================ Loss ================
        train_mean_loss, eval_loss = self._get_loss(targets, predictions)

        output_tensors[EVAL_LOSS + '_' + self.name] = eval_loss
        output_tensors[
            TRAIN_MEAN_LOSS + '_' + self.name] = train_mean_loss

        tf.summary.scalar(
            'train_mean_loss_{}'.format(self.name),
            train_mean_loss,
        )

        return train_mean_loss, eval_loss, output_tensors

    default_validation_measure = MEAN_SQUARED_ERROR

    output_config = OrderedDict([
        (LOSS, {
            'output': EVAL_LOSS,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (MEAN_SQUARED_ERROR, {
            'output': SQUARED_ERROR,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (MEAN_ABSOLUTE_ERROR, {
            'output': ABSOLUTE_ERROR,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (R2, {
            'output': R2,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (ERROR, {
            'output': ERROR,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (PREDICTIONS, {
            'output': PREDICTIONS,
            'aggregation': APPEND,
            'value': [],
            'type': PREDICTION
        })
    ])

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
        postprocessed = {}
        npy_filename = os.path.join(experiment_dir_name, '{}_{}.npy')
        name = output_feature['name']

        if PREDICTIONS in result and len(result[PREDICTIONS]) > 0:
            postprocessed[PREDICTIONS] = result[PREDICTIONS]
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PREDICTIONS),
                    result[PREDICTIONS]
                )
            del result[PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            postprocessed[PROBABILITIES] = result[PROBABILITIES]
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PROBABILITIES),
                    result[PROBABILITIES]
                )
            del result[PROBABILITIES]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature,
            LOSS,
            {'type': 'mean_squared_error', 'weight': 1}
        )
        set_default_value(output_feature[LOSS], 'type', 'mean_squared_error')
        set_default_value(output_feature[LOSS], 'weight', 1)
        set_default_value(output_feature, 'clip', None)
        set_default_value(output_feature, 'dependencies', [])
        set_default_value(output_feature, 'reduce_input', SUM)
        set_default_value(output_feature, 'reduce_dependencies', SUM)
