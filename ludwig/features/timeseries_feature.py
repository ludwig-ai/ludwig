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
import functools
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import \
    MeanAbsoluteError as MeanAbsoluteErrorMetric
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric
from tensorflow.keras.metrics import \
    RootMeanSquaredError as RootMeanSquaredErrorMetric

from ludwig.constants import *
from ludwig.encoders.sequence_encoders import StackedCNN, ParallelCNN, \
    StackedParallelCNN, StackedRNN, StackedCNNRNN, SequencePassthroughEncoder, \
    StackedTransformer
from ludwig.decoders.generic_decoders import Projector
from ludwig.features.base_feature import OutputFeature
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.feature_transform_utils import numeric_transformation_registry
from ludwig.modules.loss_modules import MSELoss, MAELoss
from ludwig.modules.metric_modules import ErrorScore, MAEMetric, MSEMetric
from ludwig.modules.metric_modules import R2Score, \
    MeanAbsolutePercentageErrorMetric, \
    WeightedMeanAbsolutePercentageErrorMetric
from ludwig.utils.misc_utils import get_from_registry, set_default_value, \
    set_default_values
from ludwig.utils.strings_utils import tokenizer_registry

logger = logging.getLogger(__name__)


class TimeseriesFeatureMixin:
    type = TIMESERIES

    preprocessing_defaults = {
        'timeseries_length_limit': 256,
        'padding_value_strategy': FILL_WITH_CONST,
        'padding_value': 0,
        'padding': 'right',
        'tokenizer': 'space',
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 0,
        'n_steps_ahead': 1,
        'undersample_ratio': 1,
        'normalization': None
    }

    @staticmethod
    def cast_column(feature, dataset_df, backend):
        return dataset_df

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        if preprocessing_parameters['column_major']:
            numeric_transformer = get_from_registry(
                preprocessing_parameters.get('normalization', None),
                numeric_transformation_registry
            )
            return_dict = \
                numeric_transformer.fit_transform_params(column, backend)

            if preprocessing_parameters['padding_value_strategy'] == FILL_WITH_MODE:
                return_dict['computed_padding_value'] = \
                    column.value_counts().index[0]
            elif preprocessing_parameters['padding_value_strategy'] == \
                    FILL_WITH_MEAN:
                return_dict['computed_padding_value'] = column.mean()
            elif preprocessing_parameters['padding_value_strategy'] == \
                    FILL_WITH_CONST:
                return_dict['computed_padding_value'] = \
                    preprocessing_parameters['padding_value']
            return_dict['max_timeseries_length'] = \
                    preprocessing_parameters['timeseries_length_limit']

        else:
            if preprocessing_parameters['padding_value_strategy'] != FILL_WITH_CONST:
                raise ValueError('Only constant padding supported for '
                                 'tabular timeseries')

            if preprocessing_parameters.get('normalization'):
                raise ValueError(
                    'Normalization is currently unsupported for '
                    'non-column-major time series')

            column = column.astype(str)
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
            return_dict = {'max_timeseries_length': max_length}

        return return_dict

    @staticmethod
    def _build_matrix(
            timeseries,
            tokenizer_name,
            length_limit,
            padding_value,
            padding,
            n_steps_ahead,
            undersample_ratio,
            backend
    ):
        tokenizer = get_from_registry(
            tokenizer_name,
            tokenizer_registry
        )()

        # TODO: Verify input order assumptions.
        rev = 1 if padding == 'right' else -1
        ts_vectors = backend.df_engine.map_objects(
            timeseries,
            lambda ts: np.array(tokenizer(ts)).astype(
                np.float32)[-n_steps_ahead::-undersample_ratio][::rev]
        )

        max_length = backend.df_engine.compute(ts_vectors.map(len).max())
        if max_length < length_limit:
            logger.debug(
                'max length of {0}: {1} < limit: {2}'.format(
                    tokenizer_name,
                    max_length,
                    length_limit
                )
            )
        max_length = length_limit

        def pad(vector):
            padded = np.full(
                (max_length,),
                padding_value,
                dtype=np.float32
            )
            limit = min(vector.shape[0], max_length)
            if padding == 'right':
                padded[:limit] = vector[:limit]
            else:  # if padding == 'left'
                padded[max_length - limit:] = vector[:limit]
            return padded

        return backend.df_engine.map_objects(ts_vectors, pad)

    @staticmethod
    def _build_matrix_from_column(
            timeseries,
            is_input,
            length_limit,
            padding_value,
            padding,
            n_steps_ahead,
            undersample_ratio,
            backend
    ):
        max_length = len(timeseries)
        if max_length < length_limit:
            logger.debug(
                'max length: {1} < limit: {2}'.format(
                    max_length,
                    length_limit
                )
            )
        max_length = length_limit

        ts_vals = timeseries.values.astype(np.float32)
        if is_input:
            # For the history, we want the most recent point to be n_steps_ahead
            # back, and then take every undersample_ratio points until reaching
            # the max length.
            last_index = lambda i: max(0, i - n_steps_ahead)
            # Iff zeros are appended on the right, put most recent values first.
            rev = 1 if padding == 'right' else -1
            history = [np.zeros(0, dtype=np.float32) if i < n_steps_ahead else \
                ts_vals[last_index(i)::-undersample_ratio][:length_limit][::rev] \
                for i in range(len(timeseries))]
            ts_vectors = backend.df_engine.df_lib.Series(history)
        else:
            if padding != 'right':
                # For the output timeseries "label", time is assumed to
                # increase from left to right, so padded values should be
                # on the right.
                raise ValueError(
                    'Padding should be "right" for output timeseries'
                )
            # If we are predicting multiple steps from an output timeseries,
            # we append the appropriate future datapoints to the current one.
            future = [ts_vals[i::undersample_ratio][:length_limit] \
                      for i in range(len(timeseries))]
            ts_vectors = backend.df_engine.df_lib.Series(future)

        def pad(vector):
            padded = np.full(
                (max_length,),
                padding_value,
                dtype=np.float32
            )
            limit = min(vector.shape[0], max_length)
            if padding == 'right':
                padded[:limit] = vector[:limit]
            else:  # if padding == 'left'
                padded[(max_length - limit):] = vector[:limit]
            return padded

        return backend.df_engine.map_objects(ts_vectors, pad)


    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters, is_input,
                     backend):
        p = preprocessing_parameters
        if p['n_steps_ahead'] > p['timeseries_length_limit']:
            raise ValueError('History window limit is less than time delay')
        if preprocessing_parameters['column_major']:
            timeseries_data = (TimeseriesFeatureMixin._build_matrix_from_column(
                column,
                is_input,
                preprocessing_parameters['timeseries_length_limit'],
                metadata['computed_padding_value'],
                preprocessing_parameters['padding'],
                preprocessing_parameters['n_steps_ahead'],
                preprocessing_parameters['undersample_ratio'],
                backend))
        else:
            timeseries_data = (TimeseriesFeatureMixin._build_matrix(
                column,
                preprocessing_parameters['tokenizer'],
                metadata['max_timeseries_length'],
                preprocessing_parameters['padding_value'],
                preprocessing_parameters['padding'],
                preprocessing_parameters['n_steps_ahead'],
                preprocessing_parameters['undersample_ratio'],
                backend))

        return timeseries_data

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend
    ):
        proc_df[feature[PROC_COLUMN]] = TimeseriesFeatureMixin.feature_data(
            input_df[feature[COLUMN]].astype(str),
            metadata[feature[NAME]],
            preprocessing_parameters,
            feature['is_input'],
            backend
        )

        normalization_type = \
            preprocessing_parameters.get('normalization', None)

        # normalize data as required
        numeric_transformer = get_from_registry(
            normalization_type,
            numeric_transformation_registry
        )(**metadata[feature[NAME]])

        proc_df[feature[PROC_COLUMN]] = \
            numeric_transformer.transform(proc_df[feature[PROC_COLUMN]])

        return proc_df


class TimeseriesInputFeature(TimeseriesFeatureMixin, SequenceInputFeature):
    encoder = 'parallel_cnn'
    max_sequence_length = None

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature, encoder_obj=encoder_obj)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.float16 or inputs.dtype == tf.float32 or \
               inputs.dtype == tf.float64
        assert len(inputs.shape) == 2

        inputs_exp = tf.cast(inputs, dtype=tf.float32)
        encoder_output = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return encoder_output

    @classmethod
    def get_input_dtype(cls):
        return tf.float32

    def get_input_shape(self):
        return self.max_sequence_length,

    @staticmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        input_feature['max_sequence_length'] = feature_metadata.get(
            'max_timeseries_length', 1)
        input_feature['embedding_size'] = 1
        input_feature['should_embed'] = False
        input_feature['computed_padding_value'] = feature_metadata.get(
            'computed_padding_value', 0)

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(
            input_feature,
            {
                TIED: None,
                'encoder': 'parallel_cnn',
            }
        )

    TimeseriesPassthroughEncoder = functools.partial(
            SequencePassthroughEncoder, expanded_dim=2)
    encoder_registry = {
        'stacked_cnn': StackedCNN,
        'parallel_cnn': ParallelCNN,
        'stacked_parallel_cnn': StackedParallelCNN,
        'rnn': StackedRNN,
        'cnnrnn': StackedCNNRNN,
        'transformer': StackedTransformer,
        'passthrough': TimeseriesPassthroughEncoder,
        'ar': TimeseriesPassthroughEncoder,
        'null': TimeseriesPassthroughEncoder,
        'none': TimeseriesPassthroughEncoder,
        'None': TimeseriesPassthroughEncoder,
        None: TimeseriesPassthroughEncoder
    }


class TimeseriesOutputFeature(TimeseriesFeatureMixin, OutputFeature):
    decoder = 'projector'
    loss = {TYPE: MEAN_SQUARED_ERROR}
    metric_functions = {LOSS: None, MEAN_SQUARED_ERROR: None,
                        MEAN_ABSOLUTE_ERROR: None, R2: None,
                        ROOT_MEAN_SQUARED_ERROR: None}
    default_validation_metric = MEAN_SQUARED_ERROR

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
        self.metric_functions = {}
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[ERROR] = ErrorScore(name='metric_error')
        self.metric_functions[MEAN_SQUARED_ERROR] = MeanSquaredErrorMetric(
            name='metric_mse'
        )
        self.metric_functions[ROOT_MEAN_SQUARED_ERROR] = \
            RootMeanSquaredErrorMetric(name='metric_rmse')
        self.metric_functions[MEAN_ABSOLUTE_ERROR] = MeanAbsoluteErrorMetric(
            name='metric_mae'
        )
        self.metric_functions[R2] = R2Score(name='metric_r2')
        # TODO: The below metrics appear skewed when the data is already
        # pre-normalized. Also, we may want options to only use a subset
        # of metrics.
        # self.metric_functions[MEAN_ABSOLUTE_PERCENTAGE_ERROR] = \
        #     MeanAbsolutePercentageErrorMetric(name='metric_mape')
        # self.metric_functions[WEIGHTED_MEAN_ABSOLUTE_PERCENTAGE_ERROR] = \
        #     WeightedMeanAbsolutePercentageErrorMetric(name='metric_wmape')

    @classmethod
    def get_output_dtype(cls):
        return tf.float32

    def get_output_shape(self):
        return self.vector_size

    @staticmethod
    def update_config_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        output_feature['vector_size'] = feature_metadata['max_timeseries_length']

    @staticmethod
    def calculate_overall_stats(
            predictions,
            targets,
            metadata
    ):
        overall_stats = {}
        numeric_transformer = get_from_registry(
            metadata['preprocessing'].get('normalization', None),
            numeric_transformation_registry
        )(**metadata)
        preds_postprocessed = \
            numeric_transformer.inverse_transform(
                predictions[PREDICTIONS].numpy()
            )
        targets_postprocessed = \
            numeric_transformer.inverse_transform(targets)
        MAPE = MeanAbsolutePercentageErrorMetric()
        WMAPE = WeightedMeanAbsolutePercentageErrorMetric()
        MAPE.update_state(targets_postprocessed, preds_postprocessed)
        WMAPE.update_state(targets_postprocessed, preds_postprocessed)
        overall_stats['mean_absolute_percentage_error'] = MAPE.result().numpy()
        overall_stats['weighted_mean_absolute_percentage_error'] = \
            WMAPE.result().numpy()
        return overall_stats

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
            # as needed convert predictions make to original value space
            numeric_transformer = get_from_registry(
                metadata['preprocessing'].get('normalization', None),
                numeric_transformation_registry
            )(**metadata)
            postprocessed[PREDICTIONS] = \
                numeric_transformer.inverse_transform(
                    predictions[PREDICTIONS].numpy()
                )

            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PREDICTIONS),
                    predictions[PREDICTIONS]
                )
            del predictions[PREDICTIONS]

        if PROBABILITIES in predictions and len(
                predictions[PROBABILITIES]) > 0:
            postprocessed[PROBABILITIES] = predictions[PROBABILITIES].numpy()
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
                'dependencies': [],
                'reduce_input': SUM,
                'reduce_dependencies': SUM
            }
        )

    decoder_registry = {
        'projector': Projector,
        'null': Projector,
        'none': Projector,
        'None': Projector,
        None: Projector
    }
