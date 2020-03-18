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
import tensorflow.compat.v1 as tf
from tensorflow.keras.losses import Reduction

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.loss_modules import \
    absolute_loss as get_absolute_loss
from ludwig.models.modules.loss_modules import \
    mean_absolute_error as get_mean_absolute_error
from ludwig.models.modules.loss_modules import \
    mean_squared_error as get_mean_squared_error
from ludwig.models.modules.loss_modules import \
    squared_loss as get_squared_loss
from ludwig.models.modules.measure_modules import ErrorScore
from ludwig.models.modules.measure_modules import R2Score
from ludwig.models.modules.measure_modules import \
    absolute_error as get_absolute_error
from ludwig.models.modules.measure_modules import error as get_error
from ludwig.models.modules.measure_modules import r2 as get_r2
from ludwig.models.modules.measure_modules import \
    squared_error as get_squared_error
from ludwig.models.modules.numerical_encoders import NumericalPassthroughEncoder
from ludwig.utils.misc import set_default_value, get_from_registry
from ludwig.utils.misc import set_default_values

logger = logging.getLogger(__name__)


class NumericalBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = NUMERICAL

    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 0,
        'normalization': None
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        if preprocessing_parameters['normalization'] is not None:
            if preprocessing_parameters['normalization'] == 'zscore':
                return {
                    'mean': column.astype(np.float32).mean(),
                    'std': column.astype(np.float32).std()
                }
            elif preprocessing_parameters['normalization'] == 'minmax':
                return {
                    'min': column.astype(np.float32).min(),
                    'max': column.astype(np.float32).max()
                }
            else:
                logger.info(
                    'Currently zscore and minmax are the only '
                    'normalization strategies available. No {}'.format(
                        preprocessing_parameters['normalization'])
                )
                return {}
        else:
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
            np.float32).values
        if preprocessing_parameters['normalization'] is not None:
            if preprocessing_parameters['normalization'] == 'zscore':
                mean = metadata[feature['name']]['mean']
                std = metadata[feature['name']]['std']
                data[feature['name']] = (data[feature['name']] - mean) / std
            elif preprocessing_parameters['normalization'] == 'minmax':
                min_ = metadata[feature['name']]['min']
                max_ = metadata[feature['name']]['max']
                data[feature['name']] = (
                                                data[feature['name']] - min_) / (max_ - min_)


class NumericalInputFeature(NumericalBaseFeature, InputFeature, tf.keras.Model):
    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)

        self.encoder = 'passthrough'
        self.norm = None
        self.dropout = False

        encoder_parameters = self.overwrite_defaults(feature)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.get_numerical_encoder(encoder_parameters)

    def get_numerical_encoder(self, encoder_parameters):
        return get_from_registry(self.encoder, numerical_encoder_registry)(
            **encoder_parameters
        )

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.float32)
        assert len(inputs.shape) == 1

        inputs_exp = tf.expand_dims(tf.cast(inputs, tf.float32), 1)
        inputs_encoded = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return inputs_encoded

    def get_last_simension(self):
        self.encoder_obj.get_last_dimension()

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
        set_default_value(input_feature, TIED, None)


class NumericalOutputFeature(NumericalBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.loss = {'type': MEAN_SQUARED_ERROR}
        self.clip = None
        self.initializer = None
        self.regularize = True

        _ = self.overwrite_defaults(feature)

        # added for tf2
        self.loss_function = None
        self.eval_function = None
        self.measure_functions = {}

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
            logger.debug('  regression_weights: {0}'.format(weights))

            biases = tf.get_variable('biases', [1])
            logger.debug('  regression_biases: {0}'.format(biases))

            predictions = tf.reshape(
                tf.matmul(hidden, weights) + biases,
                [-1]
            )
            logger.debug('  predictions: {0}'.format(predictions))

            if self.clip is not None:
                if isinstance(self.clip, (list, tuple)) and len(self.clip) == 2:
                    predictions = tf.clip_by_value(
                        predictions,
                        self.clip[0],
                        self.clip[1]
                    )
                    logger.debug(
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

    # todo tf2 remove function
    def _get_loss(self, targets, predictions):
        with tf.variable_scope('loss_{}'.format(self.name)):
            if self.loss['type'] == 'mean_squared_error':
                train_loss = tf.losses.mean_squared_error(
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

    # todo tf2 revert to original name upon completion
    def _setup_loss_tf2(self):
        if self.loss['type'] == 'mean_squared_error':
            self.loss_function = get_mean_squared_error
            self.eval_function = get_squared_loss
        elif self.loss['type'] == 'mean_absolute_error':
            self.loss_function = get_mean_absolute_error
            self.eval_function = get_absolute_loss
        else:
            train_mean_loss = None
            train_loss = None
            raise ValueError(
                'Unsupported loss type {}'.format(self.loss['type'])
            )

    # todo: remove function after tf2 port
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

    # todo: revert to original name after tf2 port
    @staticmethod
    def _setup_measures_tf2(of):
        # todo tf2 change to object method when Graph is eliminated.

        of.measure_functions.update(
            {'error': ErrorScore(name='metric_error')}
        )
        of.measure_functions.update(
            {'mse': tf.keras.metrics.MeanSquaredError(name='metric_mse')}
        )
        of.measure_functions.update(
            {'mae': tf.keras.metrics.MeanAbsoluteError(name='metric_mae')}
        )
        of.measure_functions.update(
            {'r2': R2Score(name='metric_r2')}
        )

    def reset_measures(self):
        for of_name, measure_fn in self.measure_functions.items():
            if measure_fn is not None:
                measure_fn.reset_states()

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
        predictions = self._get_predictions(
            hidden,
            hidden_size
        )

        output_tensors[PREDICTIONS + '_' + self.name] = predictions

        # ================ Measures ================
        # todo tf2 remove code after tf2 port
        error, squared_error, absolute_error, r2 = self._get_measures(
            targets,
            predictions
        )

        output_tensors[ERROR + '_' + self.name] = error
        output_tensors[SQUARED_ERROR + '_' + self.name] = squared_error
        output_tensors[ABSOLUTE_ERROR + '_' + self.name] = absolute_error
        output_tensors[R2 + '_' + self.name] = r2
        # end of code to remove

        # todo tf2 revert to original names after tf2 port
        # self._setup_measures_tf2()

        if 'sampled' not in self.loss['type']:
            tf.summary.scalar(
                'batch_train_mean_squared_error_{}'.format(self.name),
                tf.reduce_mean(squared_error)
            )
            tf.summary.scalar(
                'batch_train_mean_absolute_error_{}'.format(self.name),
                tf.reduce_mean(absolute_error)
            )
            tf.summary.scalar(
                'batch_train_mean_r2_{}'.format(self.name),
                tf.reduce_mean(r2)
            )

        # ================ Loss ================
        train_mean_loss, eval_loss = self._get_loss(targets, predictions)  # todo tf2 remove
        self._setup_loss_tf2()

        output_tensors[EVAL_LOSS + '_' + self.name] = eval_loss
        output_tensors[
            TRAIN_MEAN_LOSS + '_' + self.name] = train_mean_loss

        tf.summary.scalar(
            'batch_train_mean_loss_{}'.format(self.name),
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

        set_default_values(
            output_feature,
            {
                'clip': None,
                'dependencies': [],
                'reduce_input': SUM,
                'reduce_dependencies': SUM
            }
        )


numerical_encoder_registry = {
    'dense': FCStack,
    'passthrough': NumericalPassthroughEncoder,
    'null': NumericalPassthroughEncoder,
    'none': NumericalPassthroughEncoder,
    'None': NumericalPassthroughEncoder,
    None: NumericalPassthroughEncoder
}
