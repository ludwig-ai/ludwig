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
from tensorflow.keras.metrics import Accuracy

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.embedding_modules import Embed
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.category_decoders import Classifier
from ludwig.models.modules.category_encoders import CategoricalEmbedEncoder
from ludwig.models.modules.category_encoders import CategoricalSparseEncoder
from ludwig.models.modules.category_encoders import CategoricalPassthroughEncoder
from ludwig.models.modules.loss_modules import mean_confidence_penalty
from ludwig.models.modules.loss_modules import sampled_softmax_cross_entropy
from ludwig.models.modules.loss_modules import weighted_softmax_cross_entropy
from ludwig.models.modules.loss_modules import SoftmaxCrossEntropyLoss
from ludwig.models.modules.loss_modules import SampledSoftmaxCrossEntropyLoss
from ludwig.models.modules.metric_modules import SoftmaxCrossEntropyMetric
from ludwig.models.modules.metric_modules import accuracy as get_accuracy
from ludwig.models.modules.metric_modules import hits_at_k as get_hits_at_k
from ludwig.utils.math_utils import int_type
from ludwig.utils.math_utils import softmax
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.misc import set_default_value
from ludwig.utils.misc import set_default_values
from ludwig.utils.strings_utils import UNKNOWN_SYMBOL
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class CategoryBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = CATEGORY

    preprocessing_defaults = {
        'most_common': 10000,
        'lowercase': False,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': UNKNOWN_SYMBOL
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        idx2str, str2idx, str2freq, _ = create_vocabulary(
            column, 'stripped',
            num_most_frequent=preprocessing_parameters['most_common'],
            lowercase=preprocessing_parameters['lowercase'],
            add_padding=False
        )
        return {
            'idx2str': idx2str,
            'str2idx': str2idx,
            'str2freq': str2freq,
            'vocab_size': len(str2idx)
        }

    @staticmethod
    def feature_data(column, metadata):
        return np.array(
            column.map(
                lambda x: (
                    metadata['str2idx'][x.strip()]
                    if x.strip() in metadata['str2idx']
                    else metadata['str2idx'][UNKNOWN_SYMBOL]
                )
            ),
            dtype=int_type(metadata['vocab_size'])
        )

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters=None
    ):
        data[feature['name']] = CategoryBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']]
        )


class CategoryInputFeature(CategoryBaseFeature, InputFeature):
    def __init__(self, feature, encoder_obj=None):
        CategoryBaseFeature.__init__(self, feature)
        InputFeature.__init__(self)

        self.vocab = []

        self.embedding_size = 50
        # todo tf2: change name from 'representation' to 'encoder`
        #           update documentation
        self.representation = 'dense'
        self.embeddings_trainable = True
        self.pretrained_embeddings = None
        self.embeddings_on_cpu = False
        self.dropout = False
        self.initializer = None
        self.regularize = True


        # _ = self.overwrite_defaults(feature)
        #
        # self.embed = Embed(
        #     vocab=self.vocab,
        #     embedding_size=self.embedding_size,
        #     representation=self.representation,
        #     embeddings_trainable=self.embeddings_trainable,
        #     pretrained_embeddings=self.pretrained_embeddings,
        #     embeddings_on_cpu=self.embeddings_on_cpu,
        #     dropout=self.dropout,
        #     initializer=self.initializer,
        #     regularize=self.regularize
        # )


        self.encoder = self.representation
        encoder_parameters = self.overwrite_defaults(feature)
        # encoder_parameters.update({'input_feature_obj': self})
        encoder_parameters.update({'vocab': self.vocab})
        encoder_parameters.update({'embedding_size': self.embedding_size})
        encoder_parameters.update({'representation': self.representation})
        encoder_parameters.update({'embeddings_trainable': self.embeddings_trainable})
        encoder_parameters.update({'pretrained_embeddings': self.pretrained_embeddings})
        encoder_parameters.update({'embeddings_on_cpu': self.embeddings_on_cpu})
        encoder_parameters.update({'dropout': self.dropout})
        encoder_parameters.update({'initializer': self.initializer})
        encoder_parameters.update({'regularize': self.regularize})

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(encoder_parameters)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.int8 or inputs.dtype == tf.int16 or \
               inputs.dtype == tf.int32 or inputs.dtype == tf.int64
        assert len(inputs.shape) == 1

        inputs_exp = inputs[:, tf.newaxis]
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
        input_feature['vocab'] = feature_metadata['idx2str']

    # def _get_input_placeholder(self):
    #     return tf.placeholder(
    #         tf.int32,
    #         shape=[None],  # None is for dealing with variable batch size
    #         name='{}_placeholder'.format(self.feature_name)
    #     )
    #
    # def build_input(
    #         self,
    #         regularizer,
    #         dropout_rate,
    #         is_training=False,
    #         **kwargs
    # ):
    #     placeholder = self._get_input_placeholder()
    #     logger.debug('  placeholder: {0}'.format(placeholder))
    #
    #     # ================ Embeddings ================
    #     embedded, embedding_size = self.embed(
    #         placeholder,
    #         regularizer,
    #         dropout_rate,
    #         is_training=is_training
    #     )
    #     logger.debug('  feature_representation: {0}'.format(
    #         embedded))
    #
    #     feature_representation = {
    #         'name': self.feature_name,
    #         'type': self.type,
    #         'representation': embedded,
    #         'size': embedding_size,
    #         'placeholder': placeholder
    #     }
    #     return feature_representation

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = {
        'dense': CategoricalEmbedEncoder,
        'sparse': CategoricalSparseEncoder,
        'passthrough': CategoricalPassthroughEncoder,
        'null': CategoricalPassthroughEncoder,
        'none': CategoricalPassthroughEncoder,
        'None': CategoricalPassthroughEncoder,
        None: CategoricalPassthroughEncoder
    }


class CategoryOutputFeature(CategoryBaseFeature, OutputFeature):
    def __init__(self, feature):
        CategoryBaseFeature.__init__(self, feature)
        OutputFeature.__init__(self, feature)

        self.loss = {'type': SOFTMAX_CROSS_ENTROPY}
        self.num_classes = 0
        self.top_k = 3
        self.initializer = None
        self.regularize = True

        self.decoder = 'classifier'
        decoder_parameters = self.overwrite_defaults(feature)
        decoder_parameters.update({'num_classes': self.num_classes})

        self.decoder_obj = self.initialize_decoder(decoder_parameters)

        self._setup_loss()
        self._setup_metrics()


    def logits(
            self,
            inputs,  # hidden
    ):
        return self.decoder_obj(inputs)

    def predictions(
            self,
            inputs, # logits
    ):
        logits = inputs[LOGITS]

        probabilities = tf.nn.softmax(
            logits,
            name='probabilities_{}'.format(self.feature_name)
        )

        predictions = tf.argmax(
            logits,
            -1,
            name='predictions_{}'.format(self.feature_name)
        )
        predictions = tf.cast(predictions, dtype=tf.int32)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            LOGITS: logits
        }

    def _setup_loss(self):
        if self.loss['type'] == 'softmax_cross_entropy':
            self.train_loss_function = SoftmaxCrossEntropyLoss(
                num_classes=self.num_classes,
                feature_loss=self.loss,
                name='train_loss'
            )
        elif self.loss['type'] == 'sampled_softmax_cross_entropy':
            self.train_loss_function = SampledSoftmaxCrossEntropyLoss(
                decoder_obj=self.decoder_obj,
                num_classes=self.num_classes,
                feature_loss=self.loss,
                name='train_loss'
            )
        else:
            raise ValueError(
                "Loss type {} is not supported. Valid values are "
                "'softmax_cross_entropy' or "
                "'sampled_softmax_cross_entropy'".format(self.loss['type'])
            )

        self.eval_loss_function = SoftmaxCrossEntropyMetric(
            num_classes=self.num_classes,
            feature_loss=self.loss,
            name='eval_loss'
        )

    def _setup_metrics(self):
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[ACCURACY] = Accuracy(
            name='metric_accuracy'
        )

    default_validation_metric = ACCURACY

    # output_config = OrderedDict([
    #     (LOSS, {
    #         'output': EVAL_LOSS,
    #         'aggregation': SUM,
    #         'value': 0,
    #         'type': METRIC
    #     }),
    #     (ACCURACY, {
    #         'output': CORRECT_PREDICTIONS,
    #         'aggregation': SUM,
    #         'value': 0,
    #         'type': METRIC
    #     }),
    #     (HITS_AT_K, {
    #         'output': HITS_AT_K,
    #         'aggregation': SUM,
    #         'value': 0,
    #         'type': METRIC
    #     }),
    #     (PREDICTIONS, {
    #         'output': PREDICTIONS,
    #         'aggregation': APPEND,
    #         'value': [],
    #         'type': PREDICTION
    #     }),
    #     (PROBABILITIES, {
    #         'output': PROBABILITIES,
    #         'aggregation': APPEND,
    #         'value': [],
    #         'type': PREDICTION
    #     })
    # ])
    #
    # def _get_output_placeholder(self):
    #     return tf.placeholder(
    #         tf.int64,
    #         [None],  # None is for dealing with variable batch size
    #         name='{}_placeholder'.format(self.feature_name)
    #     )
    #
    # def _get_predictions(
    #         self,
    #         hidden,
    #         hidden_size,
    #         regularizer=None
    # ):
    #     if not self.regularize:
    #         regularizer = None
    #
    #     with tf.variable_scope('predictions_{}'.format(self.feature_name)):
    #         initializer_obj = get_initializer(self.initializer)
    #         weights = tf.get_variable(
    #             'weights',
    #             initializer=initializer_obj([hidden_size, self.num_classes]),
    #             regularizer=regularizer
    #         )
    #         logger.debug('  class_weights: {0}'.format(weights))
    #
    #         biases = tf.get_variable(
    #             'biases',
    #             [self.num_classes]
    #         )
    #         logger.debug('  class_biases: {0}'.format(biases))
    #
    #         logits = tf.matmul(hidden, weights) + biases
    #         logger.debug('  logits: {0}'.format(logits))
    #
    #         probabilities = tf.nn.softmax(
    #             logits,
    #             name='probabilities_{}'.format(self.feature_name)
    #         )
    #         predictions = tf.argmax(
    #             logits,
    #             -1,
    #             name='predictions_{}'.format(self.feature_name)
    #         )
    #
    #         with tf.device('/cpu:0'):
    #             top_k_predictions = tf.nn.top_k(
    #                 logits,
    #                 k=self.top_k,
    #                 sorted=True,
    #                 name='top_k_predictions_{}'.format(self.feature_name)
    #             )
    #
    #     return (
    #         predictions,
    #         top_k_predictions,
    #         probabilities,
    #         logits,
    #         weights,
    #         biases
    #     )
    #
    # def _get_loss(
    #         self,
    #         targets,
    #         hidden,
    #         logits,
    #         probabilities,
    #         class_weights,
    #         class_biases
    # ):
    #     with tf.variable_scope('loss_{}'.format(self.feature_name)):
    #         if ('class_similarities' in self.loss and
    #                 self.loss['class_similarities'] is not None):
    #
    #             class_similarities = self.loss['class_similarities']
    #
    #             if (class_similarities.shape[0] != self.num_classes or
    #                     class_similarities.shape[1] != self.num_classes):
    #                 logger.info(
    #                     'Class similarities is {} while num classes is {}'.format(
    #                         class_similarities.shape,
    #                         self.num_classes
    #                     )
    #                 )
    #                 if (class_similarities.shape[0] > self.num_classes and
    #                         class_similarities.shape[1] > self.num_classes):
    #                     # keep only the first num_classes rows and columns
    #                     class_similarities = class_similarities[
    #                                          :self.num_classes,
    #                                          :self.num_classes
    #                                          ]
    #                 elif (class_similarities.shape[0] < self.num_classes and
    #                       class_similarities.shape[1] < self.num_classes):
    #                     # fill the missing parts of the matrix with 0s and 1
    #                     # on the diagonal
    #                     diag = np.diag((self.num_classes, self.num_classes))
    #                     diag[
    #                     :class_similarities.shape[0],
    #                     :class_similarities.shape[1]
    #                     ] = class_similarities
    #                     class_similarities = diag
    #
    #             class_similarities = tf.constant(
    #                 class_similarities,
    #                 dtype=tf.float32,
    #                 name='class_similarities_{}'.format(self.feature_name)
    #             )
    #             vector_labels = tf.gather(
    #                 class_similarities,
    #                 targets,
    #                 name='vector_labels_{}'.format(self.feature_name)
    #             )
    #         else:
    #             vector_labels = tf.one_hot(
    #                 targets,
    #                 self.num_classes,
    #                 name='vector_labels_{}'.format(self.feature_name)
    #             )
    #
    #         if self.loss['type'] == SAMPLED_SOFTMAX_CROSS_ENTROPY:
    #             train_loss, eval_loss = sampled_softmax_cross_entropy(
    #                 targets,
    #                 hidden,
    #                 logits,
    #                 vector_labels,
    #                 class_weights,
    #                 class_biases,
    #                 self.loss,
    #                 self.num_classes
    #             )
    #         elif self.loss['type'] == SOFTMAX_CROSS_ENTROPY:
    #             train_loss = weighted_softmax_cross_entropy(
    #                 logits,
    #                 vector_labels,
    #                 self.loss
    #             )
    #             eval_loss = train_loss
    #         else:
    #             train_mean_loss = None
    #             eval_loss = None
    #             raise ValueError(
    #                 'Unsupported loss type {}'.format(self.loss['type'])
    #             )
    #
    #         if self.loss['robust_lambda'] > 0:
    #             train_loss = ((1 - self.loss['robust_lambda']) * train_loss +
    #                           self.loss['robust_lambda'] / self.num_classes)
    #
    #         train_mean_loss = tf.reduce_mean(
    #             train_loss,
    #             name='train_mean_loss_{}'.format(self.feature_name)
    #         )
    #
    #         if self.loss['confidence_penalty'] > 0:
    #             mean_penalty = mean_confidence_penalty(
    #                 probabilities,
    #                 self.num_classes
    #             )
    #             train_mean_loss += (
    #                     self.loss['confidence_penalty'] * mean_penalty
    #             )
    #
    #     return train_mean_loss, eval_loss
    #
    # def _get_metrics(self, targets, predictions, logits):
    #     with tf.variable_scope('metrics_{}'.format(self.feature_name)):
    #         accuracy_val, correct_predictions = get_accuracy(
    #             targets,
    #             predictions,
    #             self.feature_name
    #         )
    #         hits_at_k_val, mean_hits_at_k = get_hits_at_k(
    #             targets,
    #             logits,
    #             self.top_k,
    #             self.feature_name
    #         )
    #
    #     return correct_predictions, accuracy_val, hits_at_k_val, mean_hits_at_k
    #
    # def build_output(
    #         self,
    #         hidden,
    #         hidden_size,
    #         regularizer=None,
    #         dropout_rate=None,
    #         is_training=None,
    #         **kwargs
    # ):
    #     output_tensors = {}
    #
    #     # ================ Placeholder ================
    #     targets = self._get_output_placeholder()
    #     output_tensors[self.feature_name] = targets
    #     logger.debug('  targets_placeholder: {0}'.format(targets))
    #
    #     # ================ Predictions ================
    #     outs = self._get_predictions(
    #         hidden,
    #         hidden_size,
    #         regularizer=regularizer
    #     )
    #     (
    #         predictions,
    #         top_k_predictions,
    #         probabilities,
    #         logits,
    #         class_weights,
    #         class_biases
    #     ) = outs
    #
    #     output_tensors[PREDICTIONS + '_' + self.feature_name] = predictions
    #     output_tensors[TOP_K_PREDICTIONS + '_' + self.feature_name] = top_k_predictions
    #     output_tensors[PROBABILITIES + '_' + self.feature_name] = probabilities
    #
    #     # ================ metrics ================
    #     correct_predictions, accuracy, hits_at_k, mean_hits_at_k = \
    #         self._get_metrics(targets, predictions, logits)
    #
    #     output_tensors[
    #         CORRECT_PREDICTIONS + '_' + self.feature_name
    #         ] = correct_predictions
    #     output_tensors[ACCURACY + '_' + self.feature_name] = accuracy
    #     output_tensors[HITS_AT_K + '_' + self.feature_name] = hits_at_k
    #     output_tensors[MEAN_HITS_AT_K + '_' + self.feature_name] = mean_hits_at_k
    #
    #     if 'sampled' not in self.loss['type']:
    #         tf.summary.scalar(
    #             'batch_train_accuracy_{}'.format(self.feature_name),
    #             accuracy
    #         )
    #         tf.summary.scalar(
    #             'batch_train_mean_hits_at_k_{}'.format(self.feature_name),
    #             mean_hits_at_k
    #         )
    #
    #     # ================ Loss ================
    #     train_mean_loss, eval_loss = self._get_loss(
    #         targets,
    #         hidden,
    #         logits,
    #         probabilities,
    #         class_weights,
    #         class_biases
    #     )
    #
    #     output_tensors[EVAL_LOSS + '_' + self.feature_name] = eval_loss
    #     output_tensors[TRAIN_MEAN_LOSS + '_' + self.feature_name] = train_mean_loss
    #
    #     tf.summary.scalar(
    #         'batch_train_mean_loss_{}'.format(self.feature_name),
    #         train_mean_loss
    #     )
    #
    #     return train_mean_loss, eval_loss, output_tensors

    @staticmethod
    def update_model_definition_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        output_feature['num_classes'] = feature_metadata['vocab_size']
        output_feature['top_k'] = min(
            output_feature['num_classes'],
            output_feature['top_k']
        )

        if isinstance(output_feature[LOSS]['class_weights'], (list, tuple)):
            if (len(output_feature[LOSS]['class_weights']) !=
                    output_feature['num_classes']):
                raise ValueError(
                    'The length of class_weights ({}) is not compatible with '
                    'the number of classes ({}) for feature {}. '
                    'Check the metadata JSON file to see the classes '
                    'and their order and consider there needs to be a weight '
                    'for the <UNK> class too.'.format(
                        len(output_feature[LOSS]['class_weights']),
                        output_feature['num_classes'],
                        output_feature['name']
                    )
                )

        if output_feature[LOSS]['class_similarities_temperature'] > 0:
            if 'class_similarities' in output_feature[LOSS]:
                similarities = output_feature[LOSS]['class_similarities']
                temperature = output_feature[LOSS][
                    'class_similarities_temperature']

                curr_row = 0
                first_row_length = 0
                is_first_row = True
                for row in similarities:
                    if is_first_row:
                        first_row_length = len(row)
                        is_first_row = False
                        curr_row += 1
                    else:
                        curr_row_length = len(row)
                        if curr_row_length != first_row_length:
                            raise ValueError(
                                'The length of row {} of the class_similarities '
                                'of {} is {}, different from the length of '
                                'the first row {}. All rows must have '
                                'the same length.'.format(
                                    curr_row,
                                    output_feature['name'],
                                    curr_row_length,
                                    first_row_length
                                )
                            )
                        else:
                            curr_row += 1
                all_rows_length = first_row_length

                if all_rows_length != len(similarities):
                    raise ValueError(
                        'The class_similarities matrix of {} has '
                        '{} rows and {} columns, '
                        'their number must be identical.'.format(
                            output_feature['name'],
                            len(similarities),
                            all_rows_length
                        )
                    )

                if all_rows_length != output_feature['num_classes']:
                    raise ValueError(
                        'The size of the class_similarities matrix of {} is '
                        '{}, different from the number of classe ({}). '
                        'Check the metadata JSON file to see the classes '
                        'and their order and '
                        'consider <UNK> class too.'.format(
                            output_feature['name'],
                            all_rows_length,
                            output_feature['num_classes']
                        )
                    )

                similarities = np.array(similarities, dtype=np.float32)
                for i in range(len(similarities)):
                    similarities[i, :] = softmax(
                        similarities[i, :],
                        temperature=temperature
                    )

                output_feature[LOSS]['class_similarities'] = similarities
            else:
                raise ValueError(
                    'class_similarities_temperature > 0, '
                    'but no class_similarities are provided '
                    'for feature {}'.format(output_feature['name'])
                )

        if output_feature[LOSS]['type'] == 'sampled_softmax_cross_entropy':
            output_feature[LOSS]['class_counts'] = [
                feature_metadata['str2freq'][cls]
                for cls in feature_metadata['idx2str']
            ]

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
            labels=train_set_metadata[feature_name]['idx2str']
        )
        stats['confusion_matrix'] = confusion_matrix.cm.tolist()
        stats['overall_stats'] = confusion_matrix.stats()
        stats['per_class_stats'] = confusion_matrix.per_class_stats()

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
                    metadata['idx2str'][pred] for pred in preds
                ]

            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            probs = result[PROBABILITIES]
            prob = np.amax(probs, axis=1)
            postprocessed[PROBABILITIES] = probs
            postprocessed[PROBABILITY] = prob

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PROBABILITIES), probs)
                np.save(npy_filename.format(name, PROBABILITY), probs)

            del result[PROBABILITIES]

        if ('predictions_top_k' in result and
            len(result['predictions_top_k'])) > 0:

            preds_top_k = result['predictions_top_k']
            if 'idx2str' in metadata:
                postprocessed['predictions_top_k'] = [
                    [metadata['idx2str'][pred] for pred in pred_top_k]
                    for pred_top_k in preds_top_k
                ]
            else:
                postprocessed['predictions_top_k'] = preds_top_k

            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, 'predictions_top_k'),
                    preds_top_k
                )

            del result['predictions_top_k']

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        # If Loss is not defined, set an empty dictionary
        set_default_value(output_feature, LOSS, {})

        # Populate the default values for LOSS if they aren't defined already
        set_default_values(
            output_feature[LOSS],
            {
                'type': 'softmax_cross_entropy',
                'sampler': None,
                'negative_samples': 0,
                'distortion': 1,
                'unique': False,
                'labels_smoothing': 0,
                'class_weights': 1,
                'robust_lambda': 0,
                'confidence_penalty': 0,
                'class_similarities_temperature': 0,
                'weight': 1
            }
        )

        if output_feature[LOSS]['type'] == 'sampled_softmax_cross_entropy':
            set_default_values(
                output_feature[LOSS],
                {
                    'sampler': 'log_uniform',
                    'negative_samples': 25,
                    'distortion': 0.75
                }
            )
        else:
            set_default_values(
                output_feature[LOSS],
                {
                    'sampler': None,
                    'negative_samples': 0,
                    'distortion': 1
                }
            )

        set_default_values(
            output_feature,
            {
                'top_k': 3,
                'dependencies': [],
                'reduce_input': SUM,
                'reduce_dependencies': SUM
            }
        )

    decoder_registry = {
        'classifier': Classifier,
        'null': Classifier,
        'none': Classifier,
        'None': Classifier,
        None: Classifier
    }