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
from ludwig.models.modules.initializer_modules import get_initializer

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.dense_encoders import Dense
from ludwig.models.modules.loss_modules import weighted_softmax_cross_entropy
from ludwig.models.modules.measure_modules import \
    absolute_error as get_absolute_error
from ludwig.models.modules.measure_modules import error as get_error
from ludwig.models.modules.measure_modules import r2 as get_r2
from ludwig.models.modules.measure_modules import \
    squared_error as get_squared_error
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import set_default_value

logger = logging.getLogger(__name__)


class VectorBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = VECTOR

    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ""
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {
            'preprocessing': preprocessing_parameters
        }

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        """
        Expects all the vectors to be of the same size. The vectors need to be
        whitespace delimited strings. Missing values are not handled.
        """
        if len(dataset_df) == 0:
            raise ValueError("There are no vectors in the dataset provided")

        # Convert the string of features into a numpy array
        try:
            data[feature['name']] = np.array(
                [x.split() for x in dataset_df[feature['name']]], dtype=np.double
            )
        except ValueError:
            logger.error(
                'Unable to read the vector data. Make sure that all the vectors'
                ' are of the same size and do not have missing/null values.'
            )
            raise

        # Determine vector size
        vector_size = len(data[feature['name']][0])
        if 'vector_size' in preprocessing_parameters:
            if vector_size != preprocessing_parameters['vector_size']:
                raise ValueError(
                    'The user provided value for vector size ({}) does not '
                    'match the value observed in the data: {}'.format(
                        preprocessing_parameters, vector_size
                    )
                )
        else:
            logger.warning('Observed vector size: {}'.format(vector_size))

        metadata[feature['name']]['vector_size'] = vector_size


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
        return tf.compat.v1.placeholder(
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
            self.vector_size,
            regularizer,
            dropout_rate,
            is_training=is_training
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
            input_feature[key] = feature_metadata[key]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)
        set_default_value(input_feature, 'preprocessing', {})


class VectorOutputFeature(VectorBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = VECTOR
        self.vector_size = 0

        self.loss = {'type': MEAN_SQUARED_ERROR}
        self.softmax = False

        _ = self.overwrite_defaults(feature)

    def _get_output_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.float32,
            [None, self.vector_size],
            name='{}_placeholder'.format(self.name)
        )

    def _get_measures(self, targets, predictions):

        with tf.compat.v1.variable_scope('measures_{}'.format(self.name)):
            error_val = get_error(
                targets,
                predictions,
                self.name
            )

            absolute_error_val = tf.reduce_sum(
                get_absolute_error(targets, predictions, self.name), axis=1
            )

            squared_error_val = tf.reduce_sum(
                get_squared_error(targets, predictions, self.name), axis=1
            )

            # TODO - not sure if this is correct
            r2_val = tf.reduce_sum(get_r2(targets, predictions, self.name))

        return error_val, squared_error_val, absolute_error_val, r2_val

    def vector_loss(self, targets, predictions, logits):
        with tf.compat.v1.variable_scope('loss_{}'.format(self.name)):
            if self.loss['type'] == MEAN_SQUARED_ERROR:
                train_loss = tf.reduce_sum(
                    get_squared_error(targets, predictions, self.name), axis=1
                )

            elif self.loss['type'] == MEAN_ABSOLUTE_ERROR:
                train_loss = tf.reduce_sum(
                    get_absolute_error(targets, predictions, self.name), axis=1
                )

            elif self.loss['type'] == SOFTMAX_CROSS_ENTROPY:
                train_loss = weighted_softmax_cross_entropy(
                    logits,
                    targets,
                    self.loss
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

    def build_output(
            self,
            hidden,
            hidden_size,
            regularizer=None,
            dropout_rate=None,
            is_training=None,
            **kwargs
    ):
        train_mean_loss, eval_loss, output_tensors = self.build_vector_output(
            self._get_output_placeholder(),
            hidden,
            hidden_size,
            regularizer=regularizer,
        )
        return train_mean_loss, eval_loss, output_tensors

    def build_vector_output(
            self,
            targets,
            hidden,
            hidden_size,
            regularizer=None,
    ):
        feature_name = self.name
        output_tensors = {}

        # ================ Placeholder ================
        output_tensors['{}'.format(feature_name)] = targets

        # ================ Predictions ================
        logits, logits_size, predictions = self.vector_predictions(
            hidden,
            hidden_size,
            regularizer=regularizer,
        )

        output_tensors[PREDICTIONS + '_' + feature_name] = predictions

        # ================ Measures ============
        error, squared_error, absolute_error, r2 = self._get_measures(
            targets,
            predictions
        )

        output_tensors[ERROR + '_' + self.name] = error
        output_tensors[SQUARED_ERROR + '_' + self.name] = squared_error
        output_tensors[ABSOLUTE_ERROR + '_' + self.name] = absolute_error
        output_tensors[R2 + '_' + self.name] = r2

        if 'sampled' not in self.loss['type']:
            tf.compat.v1.summary.scalar(
                'train_batch_mean_squared_error_{}'.format(self.name),
                tf.reduce_mean(squared_error)
            )
            tf.compat.v1.summary.scalar(
                'train_batch_mean_absolute_error_{}'.format(self.name),
                tf.reduce_mean(absolute_error)
            )
            tf.compat.v1.summary.scalar(
                'train_batch_mean_r2_{}'.format(self.name),
                tf.reduce_mean(r2)
            )

        # ================ Loss ================
        train_mean_loss, eval_loss = self.vector_loss(
            targets, predictions, logits
        )
        output_tensors[EVAL_LOSS + '_' + self.name] = eval_loss
        output_tensors[
            TRAIN_MEAN_LOSS + '_' + self.name] = train_mean_loss

        tf.compat.v1.summary.scalar(
            'train_mean_loss_{}'.format(self.name),
            train_mean_loss,
        )

        return train_mean_loss, eval_loss, output_tensors

    def vector_predictions(
            self,
            hidden,
            hidden_size,
            regularizer=None,
    ):
        with tf.compat.v1.variable_scope('predictions_{}'.format(self.name)):
            initializer_obj = get_initializer(self.initializer)
            weights = tf.compat.v1.get_variable(
                'weights',
                initializer=initializer_obj([hidden_size, self.vector_size]),
                regularizer=regularizer
            )
            logger.debug('  projection_weights: {0}'.format(weights))

            biases = tf.compat.v1.get_variable(
                'biases',
                [self.vector_size]
            )
            logger.debug('  projection_biases: {0}'.format(biases))

            logits = tf.matmul(hidden, weights) + biases
            logger.debug('  logits: {0}'.format(logits))

            if self.softmax:
                predictions = tf.nn.softmax(logits)
            else:
                predictions = logits

        return logits, self.vector_size, predictions

    default_validation_measure = LOSS

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

        output_feature['vector_size'] = feature_metadata['vector_size']

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
            skip_save_unprocessed_output=False,
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

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):

        set_default_value(output_feature, LOSS, {})
        set_default_value(output_feature[LOSS], 'type', MEAN_SQUARED_ERROR)
        set_default_value(output_feature[LOSS], 'weight', 1)
        set_default_value(output_feature, 'reduce_input', None)
        set_default_value(output_feature, 'reduce_dependencies', None)
        set_default_value(output_feature, 'softmax', False)
        set_default_value(output_feature, 'decoder', 'fc_stack')
        set_default_value(output_feature, 'dependencies', [])


vector_encoder_registry = {
    'fc_stack': Dense
}
