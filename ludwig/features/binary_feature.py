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
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.loss_modules import mean_confidence_penalty
from ludwig.models.modules.measure_modules import accuracy as get_accuracy
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.metrics_utils import average_precision_score
from ludwig.utils.metrics_utils import precision_recall_curve
from ludwig.utils.metrics_utils import roc_auc_score
from ludwig.utils.metrics_utils import roc_curve
from ludwig.utils.misc import set_default_value
from ludwig.utils.misc import set_default_values

logger = logging.getLogger(__name__)


class BinaryBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = BINARY

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
            preprocessing_parameters=None
    ):
        data[feature['name']] = dataset_df[feature['name']].astype(
            np.bool_).values


class BinaryInputFeature(BinaryBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        _ = self.overwrite_defaults(feature)

    def _get_input_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.bool,
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
        logger.debug('  placeholder: {0}'.format(placeholder))

        feature_representation = tf.expand_dims(
            tf.cast(placeholder, tf.float32), 1)

        logger.debug('  feature_representation: {0}'.format(
            feature_representation))

        feature_representation = {
            'name': self.name,
            'type': self.type,
            'representation': feature_representation,
            'size': 1,
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
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)


class BinaryOutputFeature(BinaryBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.threshold = 0.5

        self.initializer = None
        self.regularize = True

        self.loss = {
            'robust_lambda': 0,
            'confidence_penalty': 0,
            'positive_class_weight': 1
        }

        _ = self.overwrite_defaults(feature)

    def _get_output_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.bool,
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

        with tf.compat.v1.variable_scope('predictions_{}'.format(self.name)):
            initializer_obj = get_initializer(self.initializer)
            weights = tf.compat.v1.get_variable(
                'weights',
                initializer=initializer_obj([hidden_size, 1]),
                regularizer=regularizer
            )
            logger.debug('  regression_weights: {0}'.format(weights))

            biases = tf.compat.v1.get_variable('biases', [1])
            logger.debug('  regression_biases: {0}'.format(biases))

            logits = tf.reshape(tf.matmul(hidden, weights) + biases, [-1])
            logger.debug('  logits: {0}'.format(logits))

            probabilities = tf.nn.sigmoid(
                logits,
                name='probabilities_{}'.format(
                    self.name)
            )
            predictions = tf.greater_equal(
                probabilities,
                self.threshold,
                name='predictions_{}'.format(
                    self.name)
            )
        return predictions, probabilities, logits

    def _get_loss(self, targets, logits, probabilities):
        with tf.compat.v1.variable_scope('loss_{}'.format(self.name)):
            positive_class_weight = self.loss['positive_class_weight']
            if not positive_class_weight > 0:
                raise ValueError(
                    'positive_class_weight is {}, but has to be > 0 to ensure '
                    'that loss for positive labels '
                    'p_label=1 * log(sigmoid(p_predict)) is > 0'.format(
                        positive_class_weight))

            train_loss = tf.nn.weighted_cross_entropy_with_logits(
                targets=tf.cast(targets, tf.float32),
                logits=logits,
                pos_weight=positive_class_weight
            )

            if self.loss['robust_lambda'] > 0:
                train_loss = ((1 - self.loss['robust_lambda']) * train_loss +
                              self.loss['robust_lambda'] / 2)

            train_mean_loss = tf.reduce_mean(
                train_loss,
                name='train_mean_loss_{}'.format(
                    self.name)
            )

            if self.loss['confidence_penalty'] > 0:
                mean_penalty = mean_confidence_penalty(probabilities, 2)
                train_mean_loss += (
                        self.loss['confidence_penalty'] * mean_penalty
                )

        return train_mean_loss, train_loss

    def _get_measures(self, targets, predictions):
        with tf.compat.v1.variable_scope('measures_{}'.format(self.name)):
            accuracy, correct_predictions = get_accuracy(
                targets,
                predictions,
                self.name
            )
        return correct_predictions, accuracy

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
        logger.debug('  targets_placeholder: {0}'.format(targets))
        output_tensors[self.name] = targets

        # ================ Predictions ================
        ppl = self._get_predictions(
            hidden,
            hidden_size,
            regularizer=regularizer
        )
        predictions, probabilities, logits = ppl

        output_tensors[
            PREDICTIONS + '_' + self.name] = predictions
        output_tensors[
            PROBABILITIES + '_' + self.name] = probabilities

        # ================ Measures ================
        correct_predictions, accuracy = self._get_measures(targets, predictions)

        output_tensors[
            CORRECT_PREDICTIONS + '_' + self.name] = correct_predictions

        output_tensors[ACCURACY + '_' + self.name] = accuracy

        # ================ Loss (Binary Cross Entropy) ================
        train_mean_loss, eval_loss = self._get_loss(
            targets,
            logits,
            probabilities
        )

        output_tensors[EVAL_LOSS + '_' + self.name] = eval_loss
        output_tensors[TRAIN_MEAN_LOSS + '_' + self.name] = train_mean_loss

        tf.compat.v1.summary.scalar(
            'train_mean_loss_{}'.format(self.name),
            train_mean_loss
        )

        return train_mean_loss, eval_loss, output_tensors

    default_validation_measure = ACCURACY

    output_config = OrderedDict([
        (LOSS, {
            'output': EVAL_LOSS,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (ACCURACY, {
            'output': CORRECT_PREDICTIONS,
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
            input_feature,
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
        feature_name = output_feature['name']
        stats = test_stats[feature_name]

        confusion_matrix = ConfusionMatrix(
            dataset.get(feature_name),
            stats[PREDICTIONS],
            labels=['False', 'True']
        )
        stats['confusion_matrix'] = confusion_matrix.cm.tolist()
        stats['overall_stats'] = confusion_matrix.stats()
        stats['per_class_stats'] = confusion_matrix.per_class_stats()
        fpr, tpr, thresholds = roc_curve(
            dataset.get(feature_name),
            stats[PROBABILITIES]
        )
        stats['roc_curve'] = {
            'false_positive_rate': fpr.tolist(),
            'true_positive_rate': tpr.tolist()
        }
        stats['roc_auc_macro'] = roc_auc_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='macro'
        )
        stats['roc_auc_micro'] = roc_auc_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='micro'
        )
        ps, rs, thresholds = precision_recall_curve(
            dataset.get(feature_name),
            stats[PROBABILITIES]
        )
        stats['precision_recall_curve'] = {
            'precisions': ps.tolist(),
            'recalls': rs.tolist()
        }
        stats['average_precision_macro'] = average_precision_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='macro'
        )
        stats['average_precision_micro'] = average_precision_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='micro'
        )
        stats['average_precision_samples'] = average_precision_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='samples'
        )

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
            {
                'robust_lambda': 0,
                'confidence_penalty': 0,
                'positive_class_weight': 1,
                'weight': 1
            }
        )
        set_default_values(
            output_feature,
            {
                'threshold': 0.5,
                'dependencies': [],
                'reduce_input': SUM,
                'reduce_dependencies': SUM
            }
        )
