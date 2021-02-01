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
import os

import numpy as np

from ludwig.constants import *
from ludwig.decoders.sequence_decoders import DECODER_REGISTRY
from ludwig.encoders.sequence_encoders import ENCODER_REGISTRY as SEQUENCE_ENCODER_REGISTRY
from ludwig.encoders.text_encoders import *
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.modules.loss_modules import SampledSoftmaxCrossEntropyLoss
from ludwig.modules.loss_modules import SequenceLoss
from ludwig.modules.metric_modules import EditDistanceMetric, \
    SequenceAccuracyMetric
from ludwig.modules.metric_modules import PerplexityMetric
from ludwig.modules.metric_modules import SequenceLastAccuracyMetric
from ludwig.modules.metric_modules import SequenceLossMetric
from ludwig.modules.metric_modules import TokenAccuracyMetric
from ludwig.utils.math_utils import softmax
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.strings_utils import PADDING_SYMBOL
from ludwig.utils.strings_utils import UNKNOWN_SYMBOL
from ludwig.utils.strings_utils import build_sequence_matrix
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class SequenceFeatureMixin(object):
    type = SEQUENCE

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
        'fill_value': UNKNOWN_SYMBOL
    }

    @staticmethod
    def cast_column(feature, dataset_df, backend):
        return dataset_df

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        column = column.astype(str)
        idx2str, str2idx, str2freq, max_length, _, _, _ = create_vocabulary(
            column, preprocessing_parameters['tokenizer'],
            lowercase=preprocessing_parameters['lowercase'],
            num_most_frequent=preprocessing_parameters['most_common'],
            vocab_file=preprocessing_parameters['vocab_file'],
            unknown_symbol=preprocessing_parameters['unknown_symbol'],
            padding_symbol=preprocessing_parameters['padding_symbol'],
            processor=backend.df_engine
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
    def feature_data(column, metadata, preprocessing_parameters, backend):
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
            processor=backend.df_engine
        )
        return sequence_data

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend
    ):
        sequence_data = SequenceInputFeature.feature_data(
            input_df[feature[COLUMN]].astype(str),
            metadata[feature[NAME]], preprocessing_parameters,
            backend
        )
        proc_df[feature[PROC_COLUMN]] = sequence_data
        return proc_df


class SequenceInputFeature(SequenceFeatureMixin, InputFeature):
    encoder = 'embed'
    max_sequence_length = None

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.int8 or inputs.dtype == tf.int16 or \
               inputs.dtype == tf.int32 or inputs.dtype == tf.int64
        assert len(inputs.shape) == 2

        inputs_exp = tf.cast(inputs, dtype=tf.int32)
        inputs_mask = tf.not_equal(inputs, 0)
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
        input_feature['vocab'] = feature_metadata['idx2str']
        input_feature['max_sequence_length'] = feature_metadata[
            'max_sequence_length']

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
        set_default_value(input_feature, 'encoder', 'parallel_cnn')

    encoder_registry = SEQUENCE_ENCODER_REGISTRY


class SequenceOutputFeature(SequenceFeatureMixin, OutputFeature):
    decoder = 'generator'
    loss = {TYPE: SOFTMAX_CROSS_ENTROPY}
    metric_functions = {LOSS: None, TOKEN_ACCURACY: None,
                        SEQUENCE_ACCURACY: None, LAST_ACCURACY: None,
                        PERPLEXITY: None, EDIT_DISTANCE: None}
    default_validation_metric = LOSS
    max_sequence_length = 0
    num_classes = 0

    def __init__(self, feature):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def _setup_loss(self):
        if self.loss[TYPE] == 'softmax_cross_entropy':
            self.train_loss_function = SequenceLoss()
        elif self.loss[TYPE] == 'sampled_softmax_cross_entropy':
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
                "'sampled_softmax_cross_entropy'".format(self.loss[TYPE])
            )

        self.eval_loss_function = SequenceLossMetric()

    def _setup_metrics(self):
        self.metric_functions = {}  # needed to shadow class variable
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[TOKEN_ACCURACY] = TokenAccuracyMetric()
        self.metric_functions[SEQUENCE_ACCURACY] = SequenceAccuracyMetric()
        self.metric_functions[LAST_ACCURACY] = SequenceLastAccuracyMetric()
        self.metric_functions[PERPLEXITY] = PerplexityMetric()
        self.metric_functions[EDIT_DISTANCE] = EditDistanceMetric()

    # overrides super class OutputFeature.update_metrics() method
    def update_metrics(self, targets, predictions):
        for metric, metric_fn in self.metric_functions.items():
            if metric == LOSS or metric == PERPLEXITY:
                metric_fn.update_state(targets, predictions)
            elif metric == LAST_ACCURACY:
                metric_fn.update_state(targets, predictions[LAST_PREDICTIONS])
            else:
                metric_fn.update_state(targets, predictions[PREDICTIONS])

    def logits(
            self,
            inputs,
            target=None,
            training=None
    ):
        if training and target is not None:
            return self.decoder_obj._logits_training(
                inputs,
                target=tf.cast(target, dtype=tf.int32),
                training=training
            )
        else:
            return inputs

    def predictions(self, inputs, training=None):
        # Generator Decoder
        return self.decoder_obj._predictions_eval(inputs, training=training)

    @classmethod
    def get_output_dtype(cls):
        return tf.int32

    def get_output_shape(self):
        return self.max_sequence_length,

    @staticmethod
    def update_config_with_metadata(
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
                        output_feature[COLUMN]
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
                                    output_feature[COLUMN],
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
                            output_feature[COLUMN],
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
                            output_feature[COLUMN],
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
                    'for feature {}'.format(output_feature[COLUMN])
                )

        if output_feature[LOSS][TYPE] == 'sampled_softmax_cross_entropy':
            output_feature[LOSS]['class_counts'] = [
                feature_metadata['str2freq'][cls]
                for cls in feature_metadata['idx2str']
            ]

    @staticmethod
    def calculate_overall_stats(
            predictions,
            targets,
            train_set_metadata
    ):
        overall_stats = {}
        sequences = targets
        last_elem_sequence = sequences[np.arange(sequences.shape[0]),
                                       (sequences != 0).cumsum(1).argmax(1)]
        confusion_matrix = ConfusionMatrix(
            last_elem_sequence,
            predictions[LAST_PREDICTIONS],
            labels=train_set_metadata['idx2str']
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
        postprocessed = {}
        name = self.feature_name

        npy_filename = os.path.join(output_directory, '{}_{}.npy')
        if PREDICTIONS in result and len(result[PREDICTIONS]) > 0:
            preds = result[PREDICTIONS].numpy()
            lengths = result[LENGTHS].numpy()
            if 'idx2str' in metadata:
                postprocessed[PREDICTIONS] = [
                    [metadata['idx2str'][token]
                     if token < len(metadata['idx2str']) else UNKNOWN_SYMBOL
                     for token in [pred[i] for i in range(length)]]
                    for pred, length in
                    [(preds[j], lengths[j]) for j in range(len(preds))]
                ]
            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if LAST_PREDICTIONS in result and len(result[LAST_PREDICTIONS]) > 0:
            last_preds = result[LAST_PREDICTIONS].numpy()
            if 'idx2str' in metadata:
                postprocessed[LAST_PREDICTIONS] = [
                    metadata['idx2str'][last_pred]
                    if last_pred < len(metadata['idx2str']) else UNKNOWN_SYMBOL
                    for last_pred in last_preds
                ]
            else:
                postprocessed[LAST_PREDICTIONS] = last_preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, LAST_PREDICTIONS),
                        last_preds)

            del result[LAST_PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            probs = result[PROBABILITIES].numpy()
            if probs is not None:

                # probs should be shape [b, s, nc]
                if len(probs.shape) == 3:
                    # get probability of token in that sequence position
                    seq_probs = np.amax(probs, axis=-1)

                    # sum log probability for tokens up to sequence length
                    # create mask only tokens for sequence length
                    mask = np.arange(seq_probs.shape[-1]) \
                           < np.array(result[LENGTHS]).reshape(-1, 1)
                    log_prob = np.sum(np.log(seq_probs) * mask, axis=-1)

                    # commenting probabilities out because usually it is huge:
                    # dataset x length x classes
                    # todo: add a mechanism for letting the user decide to save it
                    postprocessed[PROBABILITIES] = seq_probs
                    postprocessed[PROBABILITY] = log_prob
                else:
                    raise ValueError(
                        'Sequence probability array should be 3-dimensional '
                        'shape, instead shape is {:d}-dimensional'
                            .format(len(probs.shape))
                    )

                if not skip_save_unprocessed_output:
                    np.save(npy_filename.format(name, PROBABILITIES), seq_probs)
                    np.save(npy_filename.format(name, PROBABILITY), log_prob)

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
                TYPE: 'softmax_cross_entropy',
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
        set_default_value(output_feature[LOSS], TYPE,
                          'softmax_cross_entropy')
        set_default_value(output_feature[LOSS], 'labels_smoothing', 0)
        set_default_value(output_feature[LOSS], 'class_weights', 1)
        set_default_value(output_feature[LOSS], 'robust_lambda', 0)
        set_default_value(output_feature[LOSS], 'confidence_penalty', 0)
        set_default_value(output_feature[LOSS],
                          'class_similarities_temperature', 0)
        set_default_value(output_feature[LOSS], 'weight', 1)

        if output_feature[LOSS][TYPE] == 'sampled_softmax_cross_entropy':
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

    decoder_registry = DECODER_REGISTRY
