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

from ludwig.constants import *
from ludwig.encoders.text_encoders import ENCODER_REGISTRY
from ludwig.features.sequence_feature import SequenceInputFeature
from ludwig.features.sequence_feature import SequenceOutputFeature
from ludwig.utils.math_utils import softmax
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.misc_utils import set_default_values
from ludwig.utils.strings_utils import PADDING_SYMBOL
from ludwig.utils.strings_utils import UNKNOWN_SYMBOL
from ludwig.utils.strings_utils import build_sequence_matrix
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class TextFeatureMixin(object):
    type = TEXT

    preprocessing_defaults = {
        'char_tokenizer': 'characters',
        'char_vocab_file': None,
        'char_sequence_length_limit': 1024,
        'char_most_common': 70,
        'word_tokenizer': 'space_punct',
        'pretrained_model_name_or_path': None,
        'word_vocab_file': None,
        'word_sequence_length_limit': 256,
        'word_most_common': 20000,
        'padding_symbol': PADDING_SYMBOL,
        'unknown_symbol': UNKNOWN_SYMBOL,
        'padding': 'right',
        'lowercase': True,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': UNKNOWN_SYMBOL
    }

    @staticmethod
    def cast_column(feature, dataset_df, backend):
        return dataset_df

    @staticmethod
    def feature_meta(column, preprocessing_parameters, backend):
        (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            char_pad_idx,
            char_pad_symbol,
            char_unk_symbol,
        ) = create_vocabulary(
            column,
            tokenizer_type='characters',
            num_most_frequent=preprocessing_parameters['char_most_common'],
            lowercase=preprocessing_parameters['lowercase'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
            pretrained_model_name_or_path=preprocessing_parameters[
                'pretrained_model_name_or_path'],
            processor=backend.df_engine
        )
        (
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len,
            word_pad_idx,
            word_pad_symbol,
            word_unk_symbol,
        ) = create_vocabulary(
            column,
            tokenizer_type=preprocessing_parameters['word_tokenizer'],
            num_most_frequent=preprocessing_parameters['word_most_common'],
            lowercase=preprocessing_parameters['lowercase'],
            vocab_file=preprocessing_parameters['word_vocab_file'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
            pretrained_model_name_or_path=preprocessing_parameters[
                'pretrained_model_name_or_path'],
            processor=backend.df_engine
        )
        return (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            char_pad_idx,
            char_pad_symbol,
            char_unk_symbol,
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len,
            word_pad_idx,
            word_pad_symbol,
            word_unk_symbol,
        )

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        column = column.astype(str)
        tf_meta = TextFeatureMixin.feature_meta(
            column, preprocessing_parameters, backend
        )
        (
            char_idx2str,
            char_str2idx,
            char_str2freq,
            char_max_len,
            char_pad_idx,
            char_pad_symbol,
            char_unk_symbol,
            word_idx2str,
            word_str2idx,
            word_str2freq,
            word_max_len,
            word_pad_idx,
            word_pad_symbol,
            word_unk_symbol,
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
            'char_pad_idx': char_pad_idx,
            'char_pad_symbol': char_pad_symbol,
            'char_unk_symbol': char_unk_symbol,
            'word_idx2str': word_idx2str,
            'word_str2idx': word_str2idx,
            'word_str2freq': word_str2freq,
            'word_vocab_size': len(word_idx2str),
            'word_max_sequence_length': word_max_len,
            'word_pad_idx': word_pad_idx,
            'word_pad_symbol': word_pad_symbol,
            'word_unk_symbol': word_unk_symbol,
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters, backend):
        char_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata['char_str2idx'],
            tokenizer_type=preprocessing_parameters['char_tokenizer'],
            length_limit=metadata['char_max_sequence_length'],
            padding_symbol=metadata['char_pad_symbol'],
            padding=preprocessing_parameters['padding'],
            unknown_symbol=metadata['char_unk_symbol'],
            lowercase=preprocessing_parameters['lowercase'],
            tokenizer_vocab_file=preprocessing_parameters[
                'char_vocab_file'
            ],
            pretrained_model_name_or_path=preprocessing_parameters[
                'pretrained_model_name_or_path'
            ],
            processor=backend.df_engine
        )
        word_data = build_sequence_matrix(
            sequences=column,
            inverse_vocabulary=metadata['word_str2idx'],
            tokenizer_type=preprocessing_parameters['word_tokenizer'],
            length_limit=metadata['word_max_sequence_length'],
            padding_symbol=metadata['word_pad_symbol'],
            padding=preprocessing_parameters['padding'],
            unknown_symbol=metadata['word_unk_symbol'],
            lowercase=preprocessing_parameters['lowercase'],
            tokenizer_vocab_file=preprocessing_parameters[
                'word_vocab_file'
            ],
            pretrained_model_name_or_path=preprocessing_parameters[
                'pretrained_model_name_or_path'
            ],
            processor=backend.df_engine
        )

        return char_data, word_data

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend
    ):
        chars_data, words_data = TextFeatureMixin.feature_data(
            input_df[feature[COLUMN]].astype(str),
            metadata[feature[NAME]],
            preprocessing_parameters,
            backend
        )
        proc_df['{}_char'.format(feature[PROC_COLUMN])] = chars_data
        proc_df['{}_word'.format(feature[PROC_COLUMN])] = words_data
        return proc_df


class TextInputFeature(TextFeatureMixin, SequenceInputFeature):
    encoder = 'parallel_cnn'
    max_sequence_length = None
    level = 'word'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature, encoder_obj=encoder_obj)
        if 'pad_idx' in feature.keys():
            self.pad_idx = feature['pad_idx']
        else:
            self.pad_idx = None

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.int8 or inputs.dtype == tf.int16 or \
               inputs.dtype == tf.int32 or inputs.dtype == tf.int64
        assert len(inputs.shape) == 2

        inputs_exp = tf.cast(inputs, dtype=tf.int32)

        if self.pad_idx is not None:
            inputs_mask = tf.not_equal(inputs, self.pad_idx)
        else:
            inputs_mask = None
        lengths = tf.reduce_sum(tf.cast(inputs_mask, dtype=tf.int32), axis=1)
        encoder_output = self.encoder_obj(
            inputs_exp, training=training, mask=inputs_mask
        )

        encoder_output[LENGTHS] = lengths
        return encoder_output

    @classmethod
    def get_input_dtype(cls):
        return tf.int32

    def get_input_shape(self):
        return None,

    @staticmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        input_feature['vocab'] = (
            feature_metadata[input_feature['level'] + '_idx2str']
        )
        input_feature['max_sequence_length'] = (
            feature_metadata[input_feature['level'] + '_max_sequence_length']
        )
        input_feature['pad_idx'] = (
            feature_metadata[input_feature['level'] + '_pad_idx']
        )
        input_feature['num_tokens'] = (
            len(feature_metadata[input_feature['level'] + '_idx2str'])
        )

    @staticmethod
    def populate_defaults(input_feature):
        set_default_values(
            input_feature,
            {
                TIED: None,
                'encoder': 'parallel_cnn',
                'level': 'word'
            }
        )

        encoder_class = get_from_registry(
            input_feature['encoder'],
            TextInputFeature.encoder_registry
        )

        if hasattr(encoder_class, 'default_params'):
            set_default_values(
                input_feature,
                encoder_class.default_params
            )

    encoder_registry = ENCODER_REGISTRY


class TextOutputFeature(TextFeatureMixin, SequenceOutputFeature):
    loss = {TYPE: SOFTMAX_CROSS_ENTROPY}
    metric_functions = {LOSS: None, TOKEN_ACCURACY: None, LAST_ACCURACY: None,
                        PERPLEXITY: None, EDIT_DISTANCE: None}
    default_validation_metric = LOSS
    max_sequence_length = 0
    num_classes = 0
    level = 'word'

    def __init__(self, feature):
        super().__init__(feature)

    @classmethod
    def get_output_dtype(cls):
        return tf.int32

    def get_output_shape(self):
        return self.max_sequence_length,

    def overall_statistics_metadata(self):
        return {'level': self.level}

    @staticmethod
    def update_config_with_metadata(
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
                    'for feature {}'.format(output_feature[COLUMN])
                )

        if output_feature[LOSS][TYPE] == 'sampled_softmax_cross_entropy':
            level_str2freq = '{}_str2freq'.format(output_feature['level'])
            level_idx2str = '{}_idx2str'.format(output_feature['level'])
            output_feature[LOSS]['class_counts'] = [
                feature_metadata[level_str2freq][cls]
                for cls in feature_metadata[level_idx2str]
            ]

    @staticmethod
    def calculate_overall_stats(
            predictions,
            targets,
            train_set_metadata,
    ):
        overall_stats = {}
        level_idx2str = '{}_{}'.format(train_set_metadata['level'], 'idx2str')

        sequences = targets
        last_elem_sequence = sequences[np.arange(sequences.shape[0]),
                                       (sequences != 0).cumsum(1).argmax(1)]
        confusion_matrix = ConfusionMatrix(
            last_elem_sequence,
            predictions[LAST_PREDICTIONS],
            labels=train_set_metadata[level_idx2str]
        )
        overall_stats['confusion_matrix'] = confusion_matrix.cm.tolist()
        overall_stats['overall_stats'] = confusion_matrix.stats()
        overall_stats['per_class_stats'] = confusion_matrix.per_class_stats()

        return overall_stats

    def postprocess_predictions(
            self,
            result,
            metadata,
            output_directory,
            skip_save_unprocessed_output=False,
    ):
        # todo: refactor to reuse SequenceOutputFeature.postprocess_predictions
        postprocessed = {}
        name = self.feature_name
        level_idx2str = '{}_{}'.format(self.level, 'idx2str')

        npy_filename = os.path.join(output_directory, '{}_{}.npy')
        if PREDICTIONS in result and len(result[PREDICTIONS]) > 0:
            preds = result[PREDICTIONS].numpy()
            if level_idx2str in metadata:
                postprocessed[PREDICTIONS] = [
                    [metadata[level_idx2str][token]
                     if token < len(
                        metadata[level_idx2str]) else UNKNOWN_SYMBOL
                     for token in pred]
                    for pred in preds
                ]
            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if LAST_PREDICTIONS in result and len(result[LAST_PREDICTIONS]) > 0:
            last_preds = result[LAST_PREDICTIONS].numpy()
            if level_idx2str in metadata:
                postprocessed[LAST_PREDICTIONS] = [
                    metadata[level_idx2str][last_pred]
                    if last_pred < len(
                        metadata[level_idx2str]) else UNKNOWN_SYMBOL
                    for last_pred in last_preds
                ]
            else:
                postprocessed[LAST_PREDICTIONS] = last_preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, LAST_PREDICTIONS),
                        last_preds)

            del result[LAST_PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            probs = result[PROBABILITIES]
            if probs is not None:
                probs = probs.numpy()

                if len(probs) > 0 and isinstance(probs[0], list):
                    prob = []
                    for i in range(len(probs)):
                        for j in range(len(probs[i])):
                            probs[i][j] = np.max(probs[i][j])
                        prob.append(np.prod(probs[i]))
                else:
                    probs = np.amax(probs, axis=-1)
                    prob = np.prod(probs, axis=-1)

                # commenting probabilities out because usually it is huge:
                # dataset x length x classes
                # todo: add a mechanism for letting the user decide to save it
                # postprocessed[PROBABILITIES] = probs
                postprocessed[PROBABILITY] = prob

                if not skip_save_unprocessed_output:
                    # commenting probabilities out, see comment above
                    # np.save(npy_filename.format(name, PROBABILITIES), probs)
                    np.save(npy_filename.format(name, PROBABILITY), prob)

            del result[PROBABILITIES]

        if LENGTHS in result:
            del result[LENGTHS]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, 'level', 'word')
        SequenceOutputFeature.populate_defaults(output_feature)
