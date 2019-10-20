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
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.sequence_feature import SequenceOutputFeature
from ludwig.utils.math_utils import softmax
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.misc import set_default_value
from ludwig.utils.misc import set_default_values
from ludwig.utils.strings_utils import PADDING_SYMBOL
from ludwig.utils.strings_utils import UNKNOWN_SYMBOL
from ludwig.utils.strings_utils import build_sequence_matrix
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class TextBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = TEXT

    preprocessing_defaults = {
        'char_tokenizer': 'characters',
        'char_vocab_file': None,
        'char_sequence_length_limit': 1024,
        'char_most_common': 70,
        'word_tokenizer': 'space_punct',
        'word_vocab_file': None,
        'word_sequence_length_limit': 256,
        'word_most_common': 20000,
        'padding_symbol': PADDING_SYMBOL,
        'unknown_symbol': UNKNOWN_SYMBOL,
        'padding': 'right',
        'lowercase': True,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ''
    }

    @staticmethod
    def feature_meta(column, preprocessing_parameters):
        (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len
        ) = create_vocabulary(
            column,
            tokenizer_type='characters',
            num_most_frequent=preprocessing_parameters['char_most_common'],
            lowercase=preprocessing_parameters['lowercase'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            padding_symbol=preprocessing_parameters['padding_symbol']
        )
        (
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len
        ) = create_vocabulary(
            column,
            tokenizer_type=preprocessing_parameters['word_tokenizer'],
            num_most_frequent=preprocessing_parameters['word_most_common'],
            lowercase=preprocessing_parameters['lowercase'],
            vocab_file=preprocessing_parameters['word_vocab_file'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
        )
        return (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len
        )

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        tf_meta = TextBaseFeature.feature_meta(
            column, preprocessing_parameters
        )
        (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len
        ) = tf_meta
        char_max_len = min(
            preprocessing_parameters['char_sequence_length_limit'],
            char_max_len
        )
        word_max_len = min(
            preprocessing_parameters['word_sequence_length_limit'],
            word_max_len
        )
        return {
            'char_idx2str': char_idx2str,
            'char_str2idx': char_str2idx,
            'char_str2freq': char_str2freq,
            'char_vocab_size': len(char_idx2str),
            'char_max_sequence_length': char_max_len,
            'word_idx2str': word_idx2str,
            'word_str2idx': word_str2idx,
            'word_str2freq': word_str2freq,
            'word_vocab_size': len(word_idx2str),
            'word_max_sequence_length': word_max_len
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters):
        char_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata['char_str2idx'],
            tokenizer_type=preprocessing_parameters['char_tokenizer'],
            length_limit=metadata['char_max_sequence_length'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
            padding=preprocessing_parameters['padding'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            lowercase=preprocessing_parameters['lowercase'],
            tokenizer_vocab_file=preprocessing_parameters[
                'char_vocab_file'
            ],
        )
        word_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata['word_str2idx'],
            tokenizer_type=preprocessing_parameters['word_tokenizer'],
            length_limit=metadata['word_max_sequence_length'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
            padding=preprocessing_parameters['padding'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            lowercase=preprocessing_parameters['lowercase'],
            tokenizer_vocab_file=preprocessing_parameters[
                'word_vocab_file'
            ],
        )

        return char_data, word_data

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters
    ):
        chars_data, words_data = TextBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']], preprocessing_parameters
        )
        data['{}_char'.format(feature['name'])] = chars_data
        data['{}_word'.format(feature['name'])] = words_data


class TextInputFeature(TextBaseFeature, SequenceInputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.type = TEXT

        self.level = 'word'
        self.length = 0

        self.encoder = 'parallel_cnn'

        encoder_parameters = self.overwrite_defaults(feature)
        self.encoder_obj = self.get_sequence_encoder(encoder_parameters)

    def _get_input_placeholder(self):
        return tf.compat.v1.placeholder(
            tf.int32, shape=[None, None],
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
        logger.debug('  targets_placeholder: {0}'.format(placeholder))

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
        input_feature['vocab'] = (
            feature_metadata[input_feature['level'] + '_idx2str']
        )
        input_feature['length'] = (
            feature_metadata[input_feature['level'] + '_max_sequence_length']
        )

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(
            input_feature,
            {
                'tied_weights': None,
                'encoder': 'parallel_cnn',
                'level': 'word'
            }
        )


class TextOutputFeature(TextBaseFeature, SequenceOutputFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = TEXT

        self.level = 'word'
        self.decoder = 'generator'
        self.max_sequence_length = 0
        self.loss = {
            'type': SOFTMAX_CROSS_ENTROPY,
            'class_weights': 1,
            'class_similarities_temperature': 0,
            'weight': 1
        }
        self.num_classes = 0

        a = self.overwrite_defaults(feature)

        self.decoder_obj = self.get_sequence_decoder(feature)

    def _get_output_placeholder(self):
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
            'aggregation': SUM,
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
        output_feature['num_classes'] = feature_metadata[
            '{}_vocab_size'.format(output_feature['level'])
        ]
        output_feature['max_sequence_length'] = feature_metadata[
            '{}_max_sequence_length'.format(output_feature['level'])
        ]
        if isinstance(output_feature[LOSS]['class_weights'], (list, tuple)):
            # [0, 0] for UNK and PAD
            output_feature[LOSS]['class_weights'] = (
                    [0, 0] + output_feature[LOSS]['class_weights']
            )
            if (len(output_feature[LOSS]['class_weights']) !=
                    output_feature['num_classes']):
                raise ValueError(
                    'The length of class_weights ({}) is not compatible with '
                    'the number of classes ({})'.format(
                        len(output_feature[LOSS]['class_weights']),
                        output_feature['num_classes']
                    )
                )

        if output_feature[LOSS]['class_similarities_temperature'] > 0:
            if 'class_similarities' in output_feature:
                distances = output_feature['class_similarities']
                temperature = output_feature[LOSS][
                    'class_similarities_temperature']
                for i in range(len(distances)):
                    distances[i, :] = softmax(
                        distances[i, :],
                        temperature=temperature
                    )
                output_feature[LOSS]['class_similarities'] = distances
            else:
                raise ValueError(
                    'class_similarities_temperature > 0,'
                    'but no class similarities are provided '
                    'for feature {}'.format(output_feature['name'])
                )

        if output_feature[LOSS]['type'] == 'sampled_softmax_cross_entropy':
            level_str2freq = '{}_str2freq'.format(output_feature['level'])
            level_idx2str = '{}_idx2str'.format(output_feature['level'])
            output_feature[LOSS]['class_counts'] = [
                feature_metadata[level_str2freq][cls]
                for cls in feature_metadata[level_idx2str]
            ]

    @staticmethod
    def calculate_overall_stats(
            test_stats,
            output_feature,
            dataset,
            train_set_metadata
    ):
        feature_name = output_feature['name']
        level_idx2str = '{}_{}'.format(output_feature['level'], 'idx2str')
        sequences = dataset.get(feature_name)
        last_elem_sequence = sequences[np.arange(sequences.shape[0]),
                                       (sequences != 0).cumsum(1).argmax(1)]
        stats = test_stats[feature_name]
        confusion_matrix = ConfusionMatrix(
            last_elem_sequence,
            stats[LAST_PREDICTIONS],
            labels=train_set_metadata[feature_name][level_idx2str]
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
        level_idx2str = '{}_{}'.format(output_feature['level'], 'idx2str')

        if PREDICTIONS in result and len(result[PREDICTIONS]) > 0:
            preds = result[PREDICTIONS]
            if level_idx2str in metadata:
                postprocessed[PREDICTIONS] = [
                    [metadata[level_idx2str][token] for token in pred]
                    for pred in preds
                ]
            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if LAST_PREDICTIONS in result and len(result[LAST_PREDICTIONS]) > 0:
            last_preds = result[LAST_PREDICTIONS]
            if level_idx2str in metadata:
                postprocessed[LAST_PREDICTIONS] = [
                    metadata[level_idx2str][last_pred] for last_pred in
                    last_preds
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
                        for j in range(len(probs[i])):
                            probs[i][j] = np.max(probs[i][j])
                        prob.append(np.prod(probs[i]))
                else:
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
        set_default_value(output_feature, 'level', 'word')

        # If Loss is not defined, set an empty dictionary
        set_default_value(output_feature, LOSS, {})

        # Populate the default values for LOSS if they aren't defined already
        set_default_values(
            output_feature[LOSS],
            {
                'type': 'softmax_cross_entropy',
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

        set_default_value(output_feature[LOSS], 'unique', False)
        set_default_value(output_feature, 'decoder', 'generator')

        if output_feature['decoder'] == 'tagger':
            set_default_value(output_feature, 'reduce_input', None)

        set_default_values(
            output_feature,
            {
                'dependencies': [],
                'reduce_input': SUM,
                'reduce_dependencies': SUM
            }
        )
