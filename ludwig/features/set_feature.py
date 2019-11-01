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

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.features.feature_utils import set_str_to_idx
from ludwig.models.modules.embedding_modules import EmbedSparse
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.utils.misc import set_default_value
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class SetBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = IMAGE

    preprocessing_defaults = {
        'tokenizer': 'space',
        'most_common': 10000,
        'lowercase': False,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ''
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        idx2str, str2idx, str2freq, max_size = create_vocabulary(
            column,
            preprocessing_parameters['tokenizer'],
            num_most_frequent=preprocessing_parameters['most_common'],
            lowercase=preprocessing_parameters['lowercase']
        )
        return {
            'idx2str': idx2str,
            'str2idx': str2idx,
            'str2freq': str2freq,
            'vocab_size': len(str2idx),
            'max_set_size': max_size
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters):
        feature_vector = np.array(
            column.map(
                lambda x: set_str_to_idx(
                    x,
                    metadata['str2idx'],
                    preprocessing_parameters['tokenizer']
                )
            )
        )

        set_matrix = np.zeros(
            (len(column),
             len(metadata['str2idx'])),
            dtype=bool
        )

        for i in range(len(column)):
            set_matrix[i, feature_vector[i]] = 1

        return set_matrix

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters,
    ):
        data[feature['name']] = SetBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']],
            preprocessing_parameters
        )


class SetInputFeature(SetBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.vocab = []
        self.embedding_size = 50
        self.representation = 'dense'
        self.embeddings_trainable = True
        self.pretrained_embeddings = None
        self.embeddings_on_cpu = False
        self.dropout = False
        self.initializer = None
        self.regularize = True

        _ = self.overwrite_defaults(feature)

        self.embed_sparse = EmbedSparse(
            self.vocab,
            self.embedding_size,
            representation=self.representation,
            embeddings_trainable=self.embeddings_trainable,
            pretrained_embeddings=self.pretrained_embeddings,
            embeddings_on_cpu=self.embeddings_on_cpu,
            dropout=self.dropout,
            initializer=self.initializer,
            regularize=self.regularize
        )

    def _get_input_placeholder(self):
        # None is for dealing with variable batch size
        return tf.compat.v1.placeholder(
            tf.int32,
            shape=[None, len(self.vocab)],
            name=self.name
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

        embedded, embedding_size = self.embed_sparse(
            placeholder,
            regularizer,
            dropout_rate,
            is_training=is_training
        )
        logger.debug('  feature_representation: {0}'.format(embedded))

        feature_representation = {
            'name': self.name,
            'type': self.type,
            'representation': embedded,
            'size': embedding_size,
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
        input_feature['vocab'] = feature_metadata['idx2str']

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)


class SetOutputFeature(SetBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = SET

        self.loss = {'type': 'sigmoid_cross_entropy'}
        self.num_classes = 0
        self.threshold = 0.5
        self.initializer = None
        self.regularize = True

        _ = self.overwrite_defaults(feature)

    def _get_output_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.bool,
            shape=[None, self.num_classes],
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

        with tf.compat.v1.variable_scope('predictions_{}'.format(self.name)):
            initializer_obj = get_initializer(self.initializer)
            weights = tf.compat.v1.get_variable(
                'weights',
                initializer=initializer_obj([hidden_size, self.num_classes]),
                regularizer=regularizer
            )
            logger.debug('  class_weights: {0}'.format(weights))

            biases = tf.compat.v1.get_variable(
                'biases',
                [self.num_classes]
            )
            logger.debug('  class_biases: {0}'.format(biases))

            logits = tf.matmul(hidden, weights) + biases
            logger.debug('  logits: {0}'.format(logits))

            probabilities = tf.nn.sigmoid(
                logits,
                name='probabilities_{}'.format(self.name)
            )

            predictions = tf.greater_equal(
                probabilities,
                self.threshold,
                name='predictions_{}'.format(self.name)
            )

        return predictions, probabilities, logits

    def _get_loss(
            self,
            targets,
            logits
    ):
        with tf.compat.v1.variable_scope('loss_{}'.format(self.name)):
            train_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(targets, tf.float32),
                logits=logits
            )
            train_loss = tf.reduce_sum(train_loss, axis=1)

            train_mean_loss = tf.reduce_mean(
                train_loss,
                name='train_mean_loss_{}'.format(self.name)
            )

        return train_mean_loss, train_loss

    def _get_measures(self, targets, predictions):
        intersection = tf.reduce_sum(
            tf.cast(tf.logical_and(targets, predictions), tf.float32),
            axis=1
        )
        union = tf.reduce_sum(
            tf.cast(tf.logical_or(targets, predictions), tf.float32),
            axis=1
        )
        jaccard_index = intersection / union

        return jaccard_index

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
        ppl = self._get_predictions(
            hidden,
            hidden_size,
            regularizer=regularizer
        )
        predictions, probabilities, logits = ppl

        jaccard_index = self._get_measures(targets, predictions)

        output_tensors[PREDICTIONS + '_' + self.name] = predictions
        output_tensors[PROBABILITIES + '_' + self.name] = probabilities
        output_tensors[JACCARD + '_' + self.name] = jaccard_index

        # ================ Loss (Binary Cross Entropy) ================
        train_mean_loss, eval_loss = self._get_loss(targets, logits)

        output_tensors[EVAL_LOSS + '_' + self.name] = eval_loss
        output_tensors[TRAIN_MEAN_LOSS + '_' + self.name] = train_mean_loss

        tf.compat.v1.summary.scalar(
            'train_mean_loss_{}'.format(self.name),
            train_mean_loss
        )

        return train_mean_loss, eval_loss, output_tensors

    default_validation_measure = JACCARD

    output_config = OrderedDict([
        (LOSS, {
            'output': EVAL_LOSS,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (JACCARD, {
            'output': JACCARD,
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
        (PROBABILITIES, {
            'output': PROBABILITIES,
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
        output_feature[LOSS]['type'] = None
        output_feature['num_classes'] = feature_metadata['vocab_size']

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
            preds = result[PREDICTIONS]
            if 'idx2str' in metadata:
                postprocessed[PREDICTIONS] = [
                    [metadata['idx2str'][i] for i, pred in enumerate(pred_set)
                     if pred == True] for pred_set in preds
                ]
            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            probs = result[PROBABILITIES]
            prob = [[prob for prob in prob_set if
                     prob >= output_feature['threshold']] for prob_set in probs]
            postprocessed[PROBABILITIES] = probs
            postprocessed['probability'] = prob

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PROBABILITIES), probs)
                np.save(npy_filename.format(name, 'probability'), probs)

            del result[PROBABILITIES]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {'weight': 1, 'type': None})
        set_default_value(output_feature[LOSS], 'weight', 1)

        set_default_value(output_feature, 'threshold', 0.5)
        set_default_value(output_feature, 'dependencies', [])
        set_default_value(output_feature, 'reduce_input', SUM)
        set_default_value(output_feature, 'reduce_dependencies', SUM)
