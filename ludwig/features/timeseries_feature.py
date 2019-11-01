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
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.sequence_feature import SequenceOutputFeature
from ludwig.models.modules.measure_modules import absolute_error
from ludwig.models.modules.measure_modules import error
from ludwig.models.modules.measure_modules import r2
from ludwig.models.modules.measure_modules import squared_error
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import set_default_value
from ludwig.utils.strings_utils import tokenizer_registry

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class TimeseriesBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = TIMESERIES

    preprocessing_defaults = {
        'timeseries_length_limit': 256,
        'padding_value': 0,
        'padding': 'right',
        'tokenizer': 'space',
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ''
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        tokenizer = get_from_registry(
            preprocessing_parameters['tokenizer'],
            tokenizer_registry
        )()
        max_length = 0
        for timeseries in column:
            processed_line = tokenizer(timeseries)
            max_length = max(max_length, len(processed_line))
        max_length = min(
            preprocessing_parameters['timeseries_length_limit'],
            max_length
        )

        return {'max_timeseries_length': max_length}

    @staticmethod
    def build_matrix(
            timeseries,
            tokenizer_name,
            length_limit,
            padding_value,
            padding='right'
    ):
        tokenizer = get_from_registry(
            tokenizer_name,
            tokenizer_registry
        )()
        max_length = 0
        ts_vectors = []
        for ts in timeseries:
            ts_vector = np.array(tokenizer(ts)).astype(np.float32)
            ts_vectors.append(ts_vector)
            if len(ts_vector) > max_length:
                max_length = len(ts_vector)

        if max_length < length_limit:
            logger.debug(
                'max length of {0}: {1} < limit: {2}'.format(
                    tokenizer_name,
                    max_length,
                    length_limit
                )
            )
        max_length = length_limit
        timeseries_matrix = np.full(
            (len(timeseries), max_length),
            padding_value,
            dtype=np.float32
        )
        for i, vector in enumerate(ts_vectors):
            limit = min(vector.shape[0], max_length)
            if padding == 'right':
                timeseries_matrix[i, :limit] = vector[:limit]
            else:  # if padding == 'left
                timeseries_matrix[i, max_length - limit:] = vector[:limit]
        return timeseries_matrix

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters):
        timeseries_data = TimeseriesBaseFeature.build_matrix(
            column,
            preprocessing_parameters['tokenizer'],
            metadata['max_timeseries_length'],
            preprocessing_parameters['padding_value'],
            preprocessing_parameters['padding'])
        return timeseries_data

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        timeseries_data = TimeseriesBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']],
            preprocessing_parameters
        )
        data[feature['name']] = timeseries_data


class TimeseriesInputFeature(TimeseriesBaseFeature, SequenceInputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.type = TIMESERIES

    def _get_input_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.length],
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
        input_feature['vocab'] = []
        input_feature['length'] = feature_metadata['max_timeseries_length']
        input_feature['embedding_size'] = 1
        input_feature['should_embed'] = False

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)


# this is still WIP
class TimeseriesOutputFeature(TimeseriesBaseFeature, SequenceOutputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = TIMESERIES

        self.decoder = 'generator'

        self.loss = {
            'weight': 1,
            'type': 'softmax_cross_entropy',
            'class_weights': 1,
            'class_similarities_temperature': 0
        }
        self.num_classes = 0

        self.overwrite_defaults(feature)

        self.decoder_obj = self.get_sequence_decoder(feature)

    def _get_output_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.float32,
            [None, self.max_sequence_length],
            name='{}_placeholder'.format(self.name)
        )

    def _get_measures(self, targets, predictions):
        with tf.compat.v1.variable_scope('measures_{}'.format(self.name)):
            error_val = error(targets, predictions, self.name)
            absolute_error_val = absolute_error(targets, predictions, self.name)
            squared_error_val = squared_error(targets, predictions, self.name)
            r2_val = r2(targets, predictions, self.name)
        return error_val, squared_error_val, absolute_error_val, r2_val

    def _get_loss(self, targets, predictions):
        with tf.compat.v1.variable_scope('loss_{}'.format(self.name)):
            if self.loss['type'] == 'mean_squared_error':
                train_loss = tf.compat.v1.losses.mean_squared_error(
                    labels=targets,
                    predictions=predictions,
                    reduction=Reduction.NONE
                )
            elif self.loss['type'] == 'mean_absolute_error':
                train_loss = tf.losses.absolute_difference(
                    labels=targets,
                    predictions=predictions,
                    reduction=Reduction.NONE
                )
            else:
                train_loss = None
                train_mean_loss = None
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
        output_tensors = {}

        # ================ Placeholder ================
        targets = self._get_output_placeholder()
        output_tensors[self.name] = targets
        logger.debug('  targets_placeholder: {0}'.format(targets))

        # ================ Predictions ================
        (
            predictions_sequence,
            predictions_sequence_scores,
            predictions_sequence_length,
            last_predictions,
            targets_sequence_length,
            last_targets,
            eval_logits,
            train_logits,
            class_weights,
            class_biases
        ) = self.sequence_predictions(
            targets,
            self.decoder_obj,
            hidden,
            hidden_size,
            regularizer=regularizer,
            is_timeseries=True
        )

        output_tensors[LAST_PREDICTIONS + '_' + self.name] = last_predictions
        output_tensors[PREDICTIONS + '_' + self.name] = predictions_sequence
        output_tensors[LENGTHS + '_' + self.name] = predictions_sequence_length

        # ================ Loss ================
        train_mean_loss, eval_loss = self._get_loss(
            targets,
            predictions_sequence
        )

        output_tensors[TRAIN_MEAN_LOSS + '_' + self.name] = train_mean_loss
        output_tensors[EVAL_LOSS + '_' + self.name] = eval_loss

        tf.compat.v1.summary.scalar(TRAIN_MEAN_LOSS + '_' + self.name, train_mean_loss)

        # ================ Measures ================
        (
            error_val,
            squared_error_val,
            absolute_error_val,
            r2_val
        ) = self._get_measures(
            targets,
            predictions_sequence
        )

        output_tensors[ERROR + '_' + self.name] = error_val
        output_tensors[SQUARED_ERROR + '_' + self.name] = squared_error_val
        output_tensors[ABSOLUTE_ERROR + '_' + self.name] = absolute_error_val
        output_tensors[R2 + '_' + self.name] = r2_val

        return train_mean_loss, eval_loss, output_tensors

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
        }),
        (LENGTHS, {
            'output': LENGTHS,
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
        output_feature['max_sequence_length'] = feature_metadata[
            'max_timeseries_length'
        ]

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
        pass

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature,
            LOSS,
            {'type': 'mean_absolute_error', 'weight': 1}
        )
        set_default_value(output_feature[LOSS], 'type', 'mean_absolute_error')
        set_default_value(output_feature[LOSS], 'weight', 1)

        set_default_value(output_feature, 'decoder', 'generator')

        if output_feature['decoder'] == 'generator':
            set_default_value(output_feature, 'cell_type', 'rnn')
            set_default_value(output_feature, 'state_size', 256)
            set_default_value(output_feature, 'embedding_size', 1)
            set_default_value(output_feature, 'attention_mechanism', None)
            if output_feature['attention_mechanism'] is not None:
                set_default_value(output_feature, 'reduce_input', None)
            set_default_value(output_feature, 'decoder', 'generator')
            set_default_value(output_feature, 'decoder', 'generator')
            set_default_value(output_feature, 'decoder', 'generator')
            set_default_value(output_feature, 'decoder', 'generator')

        if output_feature['decoder'] == 'tagger':
            if 'reduce_input' not in output_feature:
                output_feature['reduce_input'] = None

        set_default_value(output_feature, 'dependencies', [])
        set_default_value(output_feature, 'reduce_input', SUM)
        set_default_value(output_feature, 'reduce_dependencies', SUM)
