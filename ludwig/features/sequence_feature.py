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
from ludwig.models.modules.loss_modules import seq2seq_sequence_loss
from ludwig.models.modules.loss_modules import \
    sequence_sampled_softmax_cross_entropy
from ludwig.models.modules.measure_modules import accuracy
from ludwig.models.modules.measure_modules import edit_distance
from ludwig.models.modules.measure_modules import masked_accuracy
from ludwig.models.modules.measure_modules import perplexity
from ludwig.models.modules.sequence_decoders import Generator
from ludwig.models.modules.sequence_decoders import Tagger
from ludwig.models.modules.sequence_encoders import BERT
from ludwig.models.modules.sequence_encoders import CNNRNN, PassthroughEncoder
from ludwig.models.modules.sequence_encoders import EmbedEncoder
from ludwig.models.modules.sequence_encoders import ParallelCNN
from ludwig.models.modules.sequence_encoders import RNN
from ludwig.models.modules.sequence_encoders import StackedCNN
from ludwig.models.modules.sequence_encoders import StackedParallelCNN
from ludwig.utils.math_utils import softmax
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import set_default_value
from ludwig.utils.strings_utils import PADDING_SYMBOL
from ludwig.utils.strings_utils import UNKNOWN_SYMBOL
from ludwig.utils.strings_utils import build_sequence_matrix
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class SequenceBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = SEQUENCE

    preprocessing_defaults = {
        'sequence_length_limit': 256,
        'most_common': 20000,
        'padding_symbol': PADDING_SYMBOL,
        'unknown_symbol': UNKNOWN_SYMBOL,
        'padding': 'right',
        'tokenizer': 'space',
        'lowercase': False,
        'vocab_file': None,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ''
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        idx2str, str2idx, str2freq, max_length = create_vocabulary(
            column, preprocessing_parameters['tokenizer'],
            lowercase=preprocessing_parameters['lowercase'],
            num_most_frequent=preprocessing_parameters['most_common'],
            vocab_file=preprocessing_parameters['vocab_file'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
        )
        max_length = min(
            preprocessing_parameters['sequence_length_limit'],
            max_length
        )
        return {
            'idx2str': idx2str,
            'str2idx': str2idx,
            'str2freq': str2freq,
            'vocab_size': len(idx2str),
            'max_sequence_length': max_length
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters):
        sequence_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata['str2idx'],
            tokenizer_type=preprocessing_parameters['tokenizer'],
            length_limit=metadata['max_sequence_length'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
            padding=preprocessing_parameters['padding'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            lowercase=preprocessing_parameters['lowercase'],
            tokenizer_vocab_file=preprocessing_parameters[
                'vocab_file'
            ],
        )
        return sequence_data

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        sequence_data = SequenceInputFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']], preprocessing_parameters)
        data[feature['name']] = sequence_data


class SequenceInputFeature(SequenceBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.encoder = 'parallel_cnn'
        self.length = 0

        encoder_parameters = self.overwrite_defaults(feature)

        self.encoder_obj = self.get_sequence_encoder(encoder_parameters)

    def get_sequence_encoder(self, encoder_parameters):
        return get_from_registry(
            self.encoder, sequence_encoder_registry)(
            **encoder_parameters
        )

    def _get_input_placeholder(self):
        # None dimension is for dealing with variable batch size
        return tf.compat.v1.placeholder(
            tf.int32,
            shape=[None, None],
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
            is_training=is_training
        )

    def build_sequence_input(
            self,
            placeholder,
            encoder,
            regularizer,
            dropout_rate,
            is_training
    ):
        feature_representation, feature_representation_size = encoder(
            placeholder,
            regularizer=regularizer,
            dropout_rate=dropout_rate,
            is_training=is_training
        )
        logger.debug('  feature_representation: {0}'.format(
            feature_representation))

        feature_representation = {
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
        input_feature['vocab'] = feature_metadata['idx2str']
        input_feature['length'] = feature_metadata['max_sequence_length']

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, 'tied_weights', None)
        set_default_value(input_feature, 'encoder', 'parallel_cnn')


class SequenceOutputFeature(SequenceBaseFeature, OutputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = SEQUENCE

        self.decoder = 'generator'
        self.max_sequence_length = 0
        self.loss = {
            'type': SOFTMAX_CROSS_ENTROPY,
            'sampler': None,
            'negative_samples': 0,
            'distortion': 1,
            'labels_smoothing': 0,
            'class_weights': 1,
            'robust_lambda': 0,
            'confidence_penalty': 0,
            'class_similarities_temperature': 0,
            'weight': 1
        }
        self.num_classes = 0

        _ = self.overwrite_defaults(feature)

        self.decoder_obj = self.get_sequence_decoder(feature)

    def get_sequence_decoder(self, decoder_parameters):
        return get_from_registry(
            self.decoder, sequence_decoder_registry)(
            **decoder_parameters
        )

    def _get_output_placeholder(self):
        # None dimension is for dealing with variable batch size
        return tf.compat.v1.placeholder(
            tf.int32,
            [None, self.max_sequence_length],
            name='{}_placeholder'.format(self.name)
        )

    def build_output(
            self,
            hidden,
            hidden_size,
            regularizer=None,
            dropout_rate=None,
            is_training=None,
            **kwargs
    ):
        train_mean_loss, eval_loss, output_tensors = self.build_sequence_output(
            self._get_output_placeholder(),
            self.decoder_obj,
            hidden,
            hidden_size,
            regularizer=regularizer,
            kwarg=kwargs
        )
        return train_mean_loss, eval_loss, output_tensors

    def build_sequence_output(
            self,
            targets,
            decoder,
            hidden,
            hidden_size,
            regularizer=None,
            **kwargs
    ):
        feature_name = self.name
        output_tensors = {}

        # ================ Placeholder ================
        output_tensors['{}'.format(feature_name)] = targets

        # ================ Predictions ================
        (
            predictions_sequence, predictions_sequence_scores,
            predictions_sequence_length, last_predictions,
            probabilities_sequence, targets_sequence_length, last_targets,
            eval_logits, train_logits, class_weights, class_biases
        ) = self.sequence_predictions(
            targets,
            decoder,
            hidden,
            hidden_size,
            regularizer=regularizer
        )

        output_tensors[LAST_PREDICTIONS + '_' + feature_name] = last_predictions
        output_tensors[PREDICTIONS + '_' + feature_name] = predictions_sequence
        output_tensors[
            PROBABILITIES + '_' + feature_name
            ] = predictions_sequence_scores
        output_tensors[
            LENGTHS + '_' + feature_name
            ] = predictions_sequence_length

        # ================ Loss ================
        train_mean_loss, eval_loss = self.sequence_loss(
            targets,
            targets_sequence_length,
            eval_logits,
            train_logits,
            class_weights,
            class_biases
        )

        output_tensors[TRAIN_MEAN_LOSS + '_' + feature_name] = train_mean_loss
        output_tensors[EVAL_LOSS + '_' + feature_name] = eval_loss

        tf.compat.v1.summary.scalar(TRAIN_MEAN_LOSS + '_' + feature_name, train_mean_loss)

        # ================ Measures ================
        (
            correct_last_predictions, last_accuracy,
            correct_overall_predictions, token_accuracy,
            correct_rowwise_predictions, rowwise_accuracy, edit_distance_val,
            mean_edit_distance, perplexity_val
        ) = self.sequence_measures(
            targets,
            targets_sequence_length,
            last_targets,
            predictions_sequence,
            predictions_sequence_length,
            last_predictions,
            eval_loss
        )

        output_tensors[
            CORRECT_LAST_PREDICTIONS + '_' + feature_name
            ] = correct_last_predictions
        output_tensors[LAST_ACCURACY + '_' + feature_name] = last_accuracy
        output_tensors[
            CORRECT_OVERALL_PREDICTIONS + '_' + feature_name
            ] = correct_overall_predictions
        output_tensors[TOKEN_ACCURACY + '_' + feature_name] = token_accuracy
        output_tensors[
            CORRECT_ROWWISE_PREDICTIONS + '_' + feature_name
            ] = correct_rowwise_predictions
        output_tensors[ROWWISE_ACCURACY + '_' + feature_name] = rowwise_accuracy
        output_tensors[EDIT_DISTANCE + '_' + feature_name] = edit_distance_val
        output_tensors[PERPLEXITY + '_' + feature_name] = perplexity_val

        if 'sampled' not in self.loss['type']:
            tf.compat.v1.summary.scalar(
                'train_batch_last_accuracy_{}'.format(feature_name),
                last_accuracy
            )
            tf.compat.v1.summary.scalar(
                'train_batch_token_accuracy_{}'.format(feature_name),
                token_accuracy
            )
            tf.compat.v1.summary.scalar(
                'train_batch_rowwise_accuracy_{}'.format(feature_name),
                rowwise_accuracy
            )
            tf.compat.v1.summary.scalar(
                'train_batch_mean_edit_distance_{}'.format(feature_name),
                mean_edit_distance
            )

        return train_mean_loss, eval_loss, output_tensors

    def sequence_predictions(
            self,
            targets,
            decoder,
            hidden,
            hidden_size,
            regularizer=None,
            is_timeseries=False
    ):
        with tf.compat.v1.variable_scope('predictions_{}'.format(self.name)):
            decoder_output = decoder(
                dict(self.__dict__),
                targets,
                hidden,
                hidden_size,
                regularizer,
                is_timeseries=is_timeseries
            )
            if self.decoder == 'generator':
                additional = 1  # because of eos symbol
            elif self.decoder == 'tagger':
                additional = 0
            else:
                additional = 0

            (
                predictions_sequence, predictions_sequence_scores,
                predictions_sequence_length, probabilities_sequence,
                targets_sequence_length, eval_logits, train_logits,
                class_weights, class_biases
            ) = decoder_output

            last_predictions = tf.gather_nd(
                predictions_sequence,
                tf.stack(
                    [tf.range(tf.shape(predictions_sequence)[0]),
                     tf.maximum(
                         predictions_sequence_length - 1 - additional,
                         0
                     )],
                    axis=1
                )
            )

            last_targets = tf.gather_nd(
                targets,
                tf.stack(
                    [tf.range(tf.shape(predictions_sequence)[0]),
                     tf.maximum(targets_sequence_length - 1 - additional, 0)],
                    axis=1
                )
            )

        return (
            predictions_sequence,
            predictions_sequence_scores,
            predictions_sequence_length,
            last_predictions,
            probabilities_sequence,
            targets_sequence_length,
            last_targets,
            eval_logits,
            train_logits,
            class_weights,
            class_biases
        )

    def sequence_measures(
            self,
            targets,
            targets_sequence_length,
            last_targets,
            predictions_sequence,
            predictions_sequence_length,
            last_predictions,
            eval_loss
    ):
        with tf.compat.v1.variable_scope('measures_{}'.format(self.name)):
            (
                token_accuracy_val,
                overall_correct_predictions,
                rowwise_accuracy_val,
                rowwise_correct_predictions
            ) = masked_accuracy(
                targets,
                predictions_sequence,
                targets_sequence_length,
                self.name
            )
            last_accuracy_val, correct_last_predictions = accuracy(
                last_targets,
                last_predictions,
                self.name
            )
            edit_distance_val, mean_edit_distance = edit_distance(
                targets,
                targets_sequence_length,
                predictions_sequence,
                predictions_sequence_length,
                self.name
            )
            perplexity_val = perplexity(eval_loss)

        return (
            correct_last_predictions,
            last_accuracy_val,
            overall_correct_predictions,
            token_accuracy_val,
            rowwise_correct_predictions,
            rowwise_accuracy_val,
            edit_distance_val,
            mean_edit_distance,
            perplexity_val
        )

    def sequence_loss(
            self,
            targets,
            targets_sequence_length,
            eval_logits,
            train_logits,
            weights,
            biases
    ):
        # This is needed because in the case of the generator decoder the
        # first padding element is also the EOS symbol and we want to count
        # the EOS symbol for the loss otherwise the model has no incentive
        # to end the sequence.
        if self.decoder == 'generator':
            targets_sequence_length = tf.minimum(
                targets_sequence_length + 1,
                tf.shape(targets)[1]
            )
        loss = self.loss
        with tf.compat.v1.variable_scope('loss_{}'.format(self.name)):
            if loss['type'] == 'softmax_cross_entropy':
                train_loss = seq2seq_sequence_loss(
                    targets,
                    targets_sequence_length,
                    eval_logits
                )
                train_mean_loss = tf.reduce_mean(
                    train_loss,
                    name='mean_loss_{}'.format(self.name)
                )
                eval_loss = train_loss

            elif loss['type'] == 'sampled_softmax_cross_entropy':
                train_loss, eval_loss = sequence_sampled_softmax_cross_entropy(
                    targets,
                    targets_sequence_length,
                    eval_logits,
                    train_logits,
                    weights,
                    biases,
                    loss,
                    self.num_classes
                )

                train_mean_loss = tf.reduce_mean(
                    train_loss,
                    name='mean_loss_{}'.format(self.name)
                )
            else:
                train_mean_loss = None
                eval_loss = None
                raise ValueError(
                    'Unsupported loss type {}'.format(loss['type'])
                )
        return train_mean_loss, eval_loss

    default_validation_measure = LOSS

    output_config = OrderedDict([
        (LOSS, {
            'output': EVAL_LOSS,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (ACCURACY, {
            'output': CORRECT_ROWWISE_PREDICTIONS,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (TOKEN_ACCURACY, {
            'output': CORRECT_OVERALL_PREDICTIONS,
            'aggregation': SEQ_SUM,
            'value': 0,
            'type': MEASURE
        }),
        (LAST_ACCURACY, {
            'output': CORRECT_LAST_PREDICTIONS,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (PERPLEXITY, {
            'output': PERPLEXITY,
            'aggregation': AVG_EXP,
            'value': 0,
            'type': MEASURE
        }),
        (EDIT_DISTANCE, {
            'output': EDIT_DISTANCE,
            'aggregation': SUM,
            'value': 0,
            'type': MEASURE
        }),
        (LAST_PREDICTIONS, {
            'output': LAST_PREDICTIONS,
            'aggregation': APPEND,
            'value': [],
            'type': PREDICTION
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
        output_feature['num_classes'] = feature_metadata['vocab_size']
        output_feature['max_sequence_length'] = (
            feature_metadata['max_sequence_length']
        )
        if isinstance(output_feature[LOSS]['class_weights'], (list, tuple)):
            if (len(output_feature[LOSS]['class_weights']) !=
                    output_feature['num_classes']):
                raise ValueError(
                    'The length of class_weights ({}) is not compatible with '
                    'the number of classes ({}) for feature {}. '
                    'Check the metadata JSON file to see the classes '
                    'and their order and consider there needs to be a weight '
                    'for the <UNK> and <PAD> class too.'.format(
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
                        'consider <UNK> and <PAD> class too.'.format(
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
        sequences = dataset.get(feature_name)
        last_elem_sequence = sequences[np.arange(sequences.shape[0]),
                                       (sequences != 0).cumsum(1).argmax(1)]
        stats = test_stats[feature_name]
        confusion_matrix = ConfusionMatrix(
            last_elem_sequence,
            stats[LAST_PREDICTIONS],
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
                    [metadata['idx2str'][token] for token in pred]
                    for pred in preds
                ]
            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if LAST_PREDICTIONS in result and len(result[LAST_PREDICTIONS]) > 0:
            last_preds = result[LAST_PREDICTIONS]
            if 'idx2str' in metadata:
                postprocessed[LAST_PREDICTIONS] = [
                    metadata['idx2str'][last_pred] for last_pred in last_preds
                ]
            else:
                postprocessed[LAST_PREDICTIONS] = last_preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, LAST_PREDICTIONS), last_preds)

            del result[LAST_PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            probs = result[PROBABILITIES]
            if probs is not None:

                if len(probs) > 0 and isinstance(probs[0], list):
                    prob = []
                    for i in range(len(probs)):
                        # todo: should adapt for the case of beam > 1
                        for j in range(len(probs[i])):
                            probs[i][j] = np.max(probs[i][j])
                        prob.append(np.prod(probs[i]))
                elif isinstance(probs, np.ndarray):
                    if (probs.shape) == 3:  # prob of each class of each token
                        probs = np.amax(probs, axis=-1)
                    prob = np.prod(probs, axis=-1)

                postprocessed[PROBABILITIES] = probs
                postprocessed['probability'] = prob

                if not skip_save_unprocessed_output:
                    np.save(npy_filename.format(name, PROBABILITIES), probs)
                    np.save(npy_filename.format(name, 'probability'), prob)

            del result[PROBABILITIES]

        if LENGTHS in result:
            del result[LENGTHS]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature,
            LOSS,
            {
                'type': 'softmax_cross_entropy',
                'sampler': None,
                'negative_samples': 0,
                'distortion': 1,
                'labels_smoothing': 0,
                'class_weights': 1,
                'robust_lambda': 0,
                'confidence_penalty': 0,
                'class_similarities_temperature': 0,
                'weight': 1
            }
        )
        set_default_value(output_feature[LOSS], 'type', 'softmax_cross_entropy')
        set_default_value(output_feature[LOSS], 'labels_smoothing', 0)
        set_default_value(output_feature[LOSS], 'class_weights', 1)
        set_default_value(output_feature[LOSS], 'robust_lambda', 0)
        set_default_value(output_feature[LOSS], 'confidence_penalty', 0)
        set_default_value(output_feature[LOSS],
                          'class_similarities_temperature', 0)
        set_default_value(output_feature[LOSS], 'weight', 1)

        if output_feature[LOSS]['type'] == 'sampled_softmax_cross_entropy':
            set_default_value(output_feature[LOSS], 'sampler', 'log_uniform')
            set_default_value(output_feature[LOSS], 'negative_samples', 25)
            set_default_value(output_feature[LOSS], 'distortion', 0.75)
        else:
            set_default_value(output_feature[LOSS], 'sampler', None)
            set_default_value(output_feature[LOSS], 'negative_samples', 0)
            set_default_value(output_feature[LOSS], 'distortion', 1)

        set_default_value(output_feature[LOSS], 'unique', False)

        set_default_value(output_feature, 'decoder', 'generator')

        if output_feature['decoder'] == 'tagger':
            set_default_value(output_feature, 'reduce_input', None)

        set_default_value(output_feature, 'dependencies', [])
        set_default_value(output_feature, 'reduce_input', SUM)
        set_default_value(output_feature, 'reduce_dependencies', SUM)


sequence_encoder_registry = {
    'stacked_cnn': StackedCNN,
    'parallel_cnn': ParallelCNN,
    'stacked_parallel_cnn': StackedParallelCNN,
    'rnn': RNN,
    'cnnrnn': CNNRNN,
    'embed': EmbedEncoder,
    'bert': BERT,
    'passthrough': PassthroughEncoder,
    'null': PassthroughEncoder,
    'none': PassthroughEncoder,
    'None': PassthroughEncoder,
    None: PassthroughEncoder
}

sequence_decoder_registry = {
    'generator': Generator,
    'tagger': Tagger
}
