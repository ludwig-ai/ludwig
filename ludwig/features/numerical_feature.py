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
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.fully_connected_modules import FCStack
# from ludwig.models.modules.loss_modules import absolute_loss
# from ludwig.models.modules.loss_modules import mean_absolute_error
# from ludwig.models.modules.loss_modules import mean_squared_error
# from ludwig.models.modules.loss_modules import squared_loss
from ludwig.models.modules.measure_modules import ErrorScore
from ludwig.models.modules.measure_modules import R2Score
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


class NumericalInputFeature(NumericalBaseFeature, InputFeature):
    def __init__(self, feature, encoder_obj=None):
        NumericalBaseFeature.__init__(self, feature)
        InputFeature.__init__(self)

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
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.float32 or inputs.dtype == tf.float64
        assert len(inputs.shape) == 1

        inputs_exp = tf.expand_dims(tf.cast(inputs, tf.float32), 1)
        inputs_encoded = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return inputs_encoded

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
        NumericalBaseFeature.__init__(self, feature)
        OutputFeature.__init__(self, feature)

        self.loss = {'type': MEAN_SQUARED_ERROR}
        self.clip = None
        self.initializer = None
        self.regularize = True

        _ = self.overwrite_defaults(feature)

        # added for tf2
        self.loss_function = None
        self.eval_function = None
        self._setup_loss()

        self.measure_functions = {}
        self._setup_measures()

        self.decoder = Dense(
            1,
            # todo tf2: add initilization, regualrization etc.
        )

    def predictions(
            self,
            inputs,  # hidden
    ):

        predictions = self.decoder(inputs)

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
                        self.feature_name,
                        self.clip
                    )
                )

        return predictions, inputs

    def _setup_loss(self):
        if self.loss['type'] == 'mean_squared_error':
            self.train_loss_function = MeanSquaredError()
            self.eval_loss_function = MeanSquaredError()
        elif self.loss['type'] == 'mean_absolute_error':
            self.train_loss_function = MeanAbsoluteError()
            self.eval_loss_function = MeanAbsoluteError()
        else:
            raise ValueError(
                'Unsupported loss type {}'.format(self.loss['type'])
            )

    def _setup_measures(self):
        self.measure_functions.update(
            {'error': ErrorScore(name='metric_error')}
        )
        self.measure_functions.update(
            {'mse': tf.keras.metrics.MeanSquaredError(name='metric_mse')}
        )
        self.measure_functions.update(
            {'mae': tf.keras.metrics.MeanAbsoluteError(name='metric_mae')}
        )
        self.measure_functions.update(
            {'r2': R2Score(name='metric_r2')}
        )

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
