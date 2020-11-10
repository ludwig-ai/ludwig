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

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import \
    MeanAbsoluteError as MeanAbsoluteErrorMetric
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric

from ludwig.constants import *
from ludwig.decoders.generic_decoders import Regressor
from ludwig.encoders.generic_encoders import PassthroughEncoder, \
    DenseEncoder
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.modules.loss_modules import MSELoss, MAELoss
from ludwig.modules.metric_modules import ErrorScore, MAEMetric, MSEMetric
from ludwig.modules.metric_modules import R2Score
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.misc_utils import set_default_values

logger = logging.getLogger(__name__)


class NumericalFeatureMixin(object):
    type = NUMERICAL
    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 0,
        'normalization': None
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        compute = backend.processor.compute
        if preprocessing_parameters['normalization'] is not None:
            if preprocessing_parameters['normalization'] == 'zscore':
                return {
                    'mean': compute(column.astype(np.float32).mean()),
                    'std': compute(column.astype(np.float32).std())
                }
            elif preprocessing_parameters['normalization'] == 'minmax':
                return {
                    'min': compute(column.astype(np.float32).min()),
                    'max': compute(column.astype(np.float32).max())
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
            dataset,
            metadata,
            preprocessing_parameters,
            backend
    ):
        dataset[feature[NAME]] = dataset_df[feature[NAME]].astype(
            np.float32).values
        if preprocessing_parameters['normalization'] is not None:
            if preprocessing_parameters['normalization'] == 'zscore':
                mean = metadata[feature[NAME]]['mean']
                std = metadata[feature[NAME]]['std']
                dataset[feature[NAME]] = (dataset[
                                              feature[NAME]] - mean) / std
            elif preprocessing_parameters['normalization'] == 'minmax':
                min_ = metadata[feature[NAME]]['min']
                max_ = metadata[feature[NAME]]['max']
                values = dataset[feature[NAME]]
                dataset[feature[NAME]] = (values - min_) / (max_ - min_)
        return dataset


class NumericalInputFeature(NumericalFeatureMixin, InputFeature):
    encoder = 'passthrough'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.float32 or inputs.dtype == tf.float64
        assert len(inputs.shape) == 1

        inputs_exp = inputs[:, tf.newaxis]
        inputs_encoded = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return inputs_encoded

    @classmethod
    def get_input_dtype(cls):
        return tf.float32

    def get_input_shape(self):
        return ()

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

    encoder_registry = {
        'dense': DenseEncoder,
        'passthrough': PassthroughEncoder,
        'null': PassthroughEncoder,
        'none': PassthroughEncoder,
        'None': PassthroughEncoder,
        None: PassthroughEncoder
    }


class NumericalOutputFeature(NumericalFeatureMixin, OutputFeature):
    decoder = 'regressor'
    loss = {TYPE: MEAN_SQUARED_ERROR}
    metric_functions = {LOSS: None, MEAN_SQUARED_ERROR: None,
                        MEAN_ABSOLUTE_ERROR: None, R2: None}
    default_validation_metric = MEAN_SQUARED_ERROR
    clip = None

    def __init__(self, feature):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(
            self,
            inputs,  # hidden
            **kwargs
    ):
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def predictions(
            self,
            inputs,  # logits
            **kwargs
    ):
        logits = inputs[LOGITS]
        predictions = logits

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

        return {PREDICTIONS: predictions, LOGITS: logits}

    def _setup_loss(self):
        if self.loss[TYPE] == 'mean_squared_error':
            self.train_loss_function = MSELoss()
            self.eval_loss_function = MSEMetric(name='eval_loss')
        elif self.loss[TYPE] == 'mean_absolute_error':
            self.train_loss_function = MAELoss()
            self.eval_loss_function = MAEMetric(name='eval_loss')
        else:
            raise ValueError(
                'Unsupported loss type {}'.format(self.loss[TYPE])
            )

    def _setup_metrics(self):
        self.metric_functions = {}  # needed to shadow class variable
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[ERROR] = ErrorScore(name='metric_error')
        self.metric_functions[MEAN_SQUARED_ERROR] = MeanSquaredErrorMetric(
            name='metric_mse'
        )
        self.metric_functions[MEAN_ABSOLUTE_ERROR] = MeanAbsoluteErrorMetric(
            name='metric_mae'
        )
        self.metric_functions[R2] = R2Score(name='metric_r2')

    # def update_metrics(self, targets, predictions):
    #     for metric in self.metric_functions.values():
    #         metric.update_state(targets, predictions[PREDICTIONS])

    @classmethod
    def get_output_dtype(cls):
        return tf.float32

    def get_output_shape(self):
        return ()

    @staticmethod
    def update_config_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def calculate_overall_stats(
            predictions,
            targets,
            metadata
    ):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
            self,
            predictions,
            metadata,
            output_directory,
            skip_save_unprocessed_output=False
    ):
        postprocessed = {}
        name = self.feature_name

        npy_filename = os.path.join(output_directory, '{}_{}.npy')
        if PREDICTIONS in predictions and len(predictions[PREDICTIONS]) > 0:
            postprocessed[PREDICTIONS] = predictions[PREDICTIONS].numpy()
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PREDICTIONS),
                    predictions[PREDICTIONS]
                )
            del predictions[PREDICTIONS]

        if PROBABILITIES in predictions and len(
                predictions[PROBABILITIES]) > 0:
            postprocessed[PROBABILITIES] = predictions[PROBABILITIES]
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PROBABILITIES),
                    predictions[PROBABILITIES]
                )
            del predictions[PROBABILITIES]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature,
            LOSS,
            {TYPE: 'mean_squared_error', 'weight': 1}
        )
        set_default_value(output_feature[LOSS], TYPE, 'mean_squared_error')
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

    decoder_registry = {
        'regressor': Regressor,
        'null': Regressor,
        'none': Regressor,
        'None': Regressor,
        None: Regressor
    }
