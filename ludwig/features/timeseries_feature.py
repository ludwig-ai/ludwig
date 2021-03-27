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
from ludwig.encoders.sequence_encoders import StackedCNN, ParallelCNN, \
    StackedParallelCNN, StackedRNN, StackedCNNRNN, SequencePassthroughEncoder, \
    StackedTransformer
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.feature_transform_utils import numeric_transformation_registry
from ludwig.utils.misc_utils import get_from_registry, set_default_values
from ludwig.utils.strings_utils import tokenizer_registry

logger = logging.getLogger(__name__)


class TimeseriesFeatureMixin(object):
    type = TIMESERIES

    preprocessing_defaults = {
        'timeseries_length_limit': 256,
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

        if preprocessing_parameters['column_major']:
            numeric_transformer = get_from_registry(
                preprocessing_parameters.get('normalization', None),
                numeric_transformation_registry
            )
            return_dict = {**return_dict,
                **numeric_transformer.fit_transform_params(column, backend)}
        return return_dict

    @staticmethod
    def build_matrix(
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
    def build_matrix_from_column(
            timeseries,
            tokenizer_name,
            length_limit,
            padding_value,
            padding,
            n_steps_ahead,
            undersample_ratio,
            missing_value_strategy,
            backend
    ):
        max_length = len(timeseries)
        if max_length < length_limit:
            logger.debug(
                'max length of {0}: {1} < limit: {2}'.format(
                    tokenizer_name,
                    max_length,
                    length_limit
                )
            )
        max_length = length_limit
        # For the history, we want the most recent point to be n_steps_ahead
        # back, and then take every undersample_ratio points until reaching
        # the max length
        last_index = lambda i: max(0, i - n_steps_ahead)
        ts_vals = timeseries.values.astype(np.float32)
        # Iff zeros are appended on the right, put most recent values first
        rev = 1 if padding == 'right' else -1
        history = [np.zeros(0, dtype=np.float32) if i < n_steps_ahead else \
            ts_vals[last_index(i)::-undersample_ratio][:length_limit][::rev] \
            for i in range(len(timeseries))]
        ts_vectors = backend.df_engine.df_lib.Series(history)

        # Duplication of logic - not ideal. Also, need error handling.
        if missing_value_strategy == 'fill_with_mean':
            padding_value = np.mean(ts_vals)

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
    def feature_data(column, metadata, preprocessing_parameters, backend):
        p = preprocessing_parameters
        if p['n_steps_ahead'] > p['timeseries_length_limit']:
            raise ValueError('History window limit is less than time delay')
        if preprocessing_parameters['column_major']:
            timeseries_data = (TimeseriesFeatureMixin.build_matrix_from_column(
                column,
                preprocessing_parameters['tokenizer'],
                preprocessing_parameters['timeseries_length_limit'],
                preprocessing_parameters['padding_value'],
                preprocessing_parameters['padding'],
                preprocessing_parameters['n_steps_ahead'],
                preprocessing_parameters['undersample_ratio'],
                preprocessing_parameters['missing_value_strategy'],
                backend))
        else:
            timeseries_data = (TimeseriesFeatureMixin.build_matrix(
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
            backend
        )

        # normalize data as required
        numeric_transformer = get_from_registry(
            preprocessing_parameters.get('normalization', None),
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
        input_feature['max_sequence_length'] = feature_metadata[
            'max_timeseries_length']
        input_feature['embedding_size'] = 1
        input_feature['should_embed'] = False

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(
            input_feature,
            {
                TIED: None,
                'encoder': 'parallel_cnn',
            }
        )

    encoder_registry = {
        'stacked_cnn': StackedCNN,
        'parallel_cnn': ParallelCNN,
        'stacked_parallel_cnn': StackedParallelCNN,
        'rnn': StackedRNN,
        'cnnrnn': StackedCNNRNN,
        'transformer': StackedTransformer,
        'passthrough': SequencePassthroughEncoder,
        'null': SequencePassthroughEncoder,
        'none': SequencePassthroughEncoder,
        'None': SequencePassthroughEncoder,
        None: SequencePassthroughEncoder
    }
