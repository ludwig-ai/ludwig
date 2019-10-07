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

import tensorflow as tf

from ludwig.models.modules.attention_modules import \
    feed_forward_memory_attention
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.recurrent_modules import recurrent_decoder
from ludwig.utils.tf_utils import sequence_length_2D, sequence_length_3D


logger = logging.getLogger(__name__)


class Generator:
    def __init__(
            self,
            cell_type='rnn',
            state_size=256,
            embedding_size=64,
            beam_width=1,
            num_layers=1,
            attention_mechanism=None,
            tied_embeddings=None,
            initializer=None,
            regularize=True,
            **kwargs
    ):
        self.cell_type = cell_type
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.beam_width = beam_width
        self.num_layers = num_layers
        self.attention_mechanism = attention_mechanism
        self.tied_embeddings = tied_embeddings
        self.initializer = initializer
        self.regularize = regularize

    def __call__(
            self,
            output_feature,
            targets,
            hidden,
            hidden_size,
            regularizer,
            is_timeseries=False
    ):
        if len(hidden.shape) != 3 and self.attention_mechanism is not None:
            raise ValueError(
                'Encoder outputs rank is {}, but should be 3 [batch x sequence x hidden] '
                'when attention mechanism is {}. '
                'If you are using a sequential encoder or combiner consider setting reduce_output to None '
                'and flatten to False if those parameters apply.'
                'Also make sure theat reduce_input of {} output feature is None,'.format(
                    len(hidden.shape), self.attention_mechanism,
                    output_feature['name']))
        if len(hidden.shape) != 2 and self.attention_mechanism is None:
            raise ValueError(
                'Encoder outputs rank is {}, but should be 2 [batch x hidden] '
                'when attention mechanism is {}. '
                'Consider setting reduce_input of {} output feature to a value different from None.'.format(
                    len(hidden.shape), self.attention_mechanism,
                    output_feature['name']))

        if is_timeseries:
            vocab_size = 1
        else:
            vocab_size = output_feature['num_classes']

        if not self.regularize:
            regularizer = None

        tied_embeddings_tensor = None
        if self.tied_embeddings is not None:
            try:
                tied_embeddings_tensor = tf.get_default_graph().get_tensor_by_name(
                    '{}/embeddings:0'.format(self.tied_embeddings))
            except:
                raise ValueError(
                    'An error occurred while obtaining embeddings from the feature {} '
                    'to use as tied weights in the generator decoder of feature {}. '
                    '{} does not exists or does not have an embedding weights.v'
                    'Please check the spelling of the feature name '
                    'in the tied_embeddings field and '
                    'be sure its type is not binary, numerical or timeseries.'.format(
                        self.tied_embeddings,
                        output_feature['name'],
                        self.tied_embeddings
                    )
                )

        predictions_sequence, predictions_sequence_scores, \
        predictions_sequence_length_with_eos, \
        targets_sequence_length_with_eos, eval_logits, train_logits, \
        class_weights, class_biases = recurrent_decoder(
            hidden,
            targets,
            output_feature['max_sequence_length'],
            vocab_size,
            cell_type=self.cell_type,
            state_size=self.state_size,
            embedding_size=self.embedding_size,
            beam_width=self.beam_width,
            num_layers=self.num_layers,
            attention_mechanism=self.attention_mechanism,
            is_timeseries=is_timeseries,
            embeddings=tied_embeddings_tensor,
            initializer=self.initializer,
            regularizer=regularizer
        )

        probabilities_target_sequence = tf.nn.softmax(eval_logits)

        return predictions_sequence, predictions_sequence_scores, \
               predictions_sequence_length_with_eos, \
               probabilities_target_sequence, targets_sequence_length_with_eos, \
               eval_logits, train_logits, class_weights, class_biases


class Tagger:
    def __init__(
            self,
            initializer=None,
            regularize=True,
            attention=False,
            **kwargs
    ):
        self.initializer = initializer
        self.regularize = regularize
        self.attention = attention

    def __call__(
            self,
            output_feature,
            targets,
            hidden,
            hidden_size,
            regularizer,
            is_timeseries=False
    ):
        logger.debug('  hidden shape: {0}'.format(hidden.shape))
        if len(hidden.shape) != 3:
            raise ValueError(
                'Decoder inputs rank is {}, but should be 3 [batch x sequence x hidden] '
                'when using a tagger sequential decoder. '
                'Consider setting reduce_output to null / None if a sequential encoder / combiner is used.'.format(
                    len(hidden.shape)))

        if is_timeseries:
            output_feature['num_classes'] = 1

        if not self.regularize:
            regularizer = None

        sequence_length = tf.shape(hidden)[1]

        if self.attention:
            hidden, hidden_size = feed_forward_memory_attention(
                hidden,
                hidden,
                hidden_size
            )
        targets_sequence_length = sequence_length_2D(targets)

        initializer_obj = get_initializer(self.initializer)
        class_weights = tf.compat.v1.get_variable(
            'weights',
            initializer=initializer_obj(
                [hidden_size, output_feature['num_classes']]),
            regularizer=regularizer
        )
        logger.debug('  weights: {0}'.format(class_weights))

        class_biases = tf.compat.v1.get_variable(
            'biases',
            [output_feature['num_classes']]
        )
        logger.debug('  biases: {0}'.format(class_biases))

        hidden_reshape = tf.reshape(hidden, [-1, hidden_size])
        logits_to_reshape = tf.matmul(hidden_reshape,
                                      class_weights) + class_biases
        logits = tf.reshape(
            logits_to_reshape,
            [-1, sequence_length, output_feature['num_classes']]
        )
        logger.debug('  logits: {0}'.format(logits))

        if is_timeseries:
            probabilities_sequence = tf.zeros_like(logits)
            predictions_sequence = tf.reshape(logits, [-1, sequence_length])
        else:
            probabilities_sequence = tf.nn.softmax(
                logits,
                name='probabilities_{}'.format(output_feature['name'])
            )
            predictions_sequence = tf.argmax(
                logits,
                -1,
                name='predictions_{}'.format(output_feature['name']),
                output_type=tf.int32
            )

        predictions_sequence_length = sequence_length_3D(hidden)

        return predictions_sequence, probabilities_sequence, \
               predictions_sequence_length, \
               probabilities_sequence, targets_sequence_length, \
               logits, hidden, class_weights, class_biases
