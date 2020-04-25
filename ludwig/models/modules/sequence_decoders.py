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

import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, LSTMCell
import tensorflow_addons as tfa

from ludwig.models.modules.attention_modules import \
    feed_forward_memory_attention
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.recurrent_modules import recurrent_decoder

logger = logging.getLogger(__name__)


class SequenceGeneratorDecoder(Layer):
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
            is_timeseries=False,
            num_classes=0,
            **kwargs
    ):
        super(SequenceGeneratorDecoder, self).__init__()

        self.cell_type = cell_type
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.beam_width = beam_width
        self.num_layers = num_layers
        self.attention_mechanism = attention_mechanism
        self.tied_embeddings = tied_embeddings
        self.initializer = initializer
        self.regularize = regularize
        self.is_timeseries = is_timeseries
        self.num_classes = num_classes

        self.embeddings_dec = Embedding(num_classes, embedding_size)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.decoder_cell = LSTMCell(state_size)
        self.projection_layer = Dense(num_classes)
        self.decoder = \
            tfa.seq2seq.basic_decoder.BasicDecoder(self.decoder_cell,
                                                    self.sampler,
                                                    output_layer=self.projection_layer)

    # todo tf2: remove if not needed
    # def build_initial_state1(self, batch_size, encoder_state=None):
    #     initial_state = self.decoder_cell.get_initial_state(
    #         inputs=encoder_state,
    #         batch_size=batch_size,
    #         dtype=tf.float32
    #     )
    #     return initial_state

    def build_sequence_lengths(self, batch_size):
        # todo tf2 use of self.num_classes is a placeholder, need to confirm correct approach
        return np.ones((batch_size,)).astype(np.int32) * self.num_classes

    def build_initial_state(self, batch_size, state_size):
        zero_state = tf.zeros([batch_size, state_size], dtype=tf.float32)
        return [zero_state, zero_state]

    def call(
            self,
            inputs,
            training=None,
            mask=None
    ):
        if len(inputs.shape) != 3 and self.attention_mechanism is not None:
            raise ValueError(
                'Encoder outputs rank is {}, but should be 3 [batch x sequence x hidden] '
                'when attention mechanism is {}. '
                'If you are using a sequential encoder or combiner consider setting reduce_output to None '
                'and flatten to False if those parameters apply.'
                'Also make sure theat reduce_input of {} output feature is None,'.format(
                    len(inputs.shape), self.attention_mechanism,
                    self.output_feature))
        if len(inputs.shape) != 2 and self.attention_mechanism is None:
            raise ValueError(
                'Encoder outputs rank is {}, but should be 2 [batch x hidden] '
                'when attention mechanism is {}. '
                'Consider setting reduce_input of {} output feature to a value different from None.'.format(
                    len(inputs.shape), self.attention_mechanism,
                    self.output_feature))

        decoder_embeddings = self.embeddings_dec(inputs)

        sequence_lengths = self.build_sequence_lengths(inputs.shape[0])

        initial_state = self.build_initial_state(inputs.shape[0],
                                                 self.state_size)

        final_outputs, final_state, final_sequence_lengths = self.decoder(
            decoder_embeddings, initial_state=initial_state,
            sequence_length=sequence_lengths)

        return final_outputs.rnn_output  # todo tf2 in case we need, final_outputs, final_state, final_sequence_lengths


        # TODO TF2 clean up after port
        # tied_embeddings_tensor = None
        # todo tf2  determine how to handle following
        # if self.tied_embeddings is not None:
        #     try:
        #         tied_embeddings_tensor = tf.get_default_graph().get_tensor_by_name(
        #             '{}/embeddings:0'.format(self.tied_embeddings))
        #     except:
        #         raise ValueError(
        #             'An error occurred while obtaining embeddings from the feature {} '
        #             'to use as tied weights in the generator decoder of feature {}. '
        #             '{} does not exists or does not have an embedding weights.v'
        #             'Please check the spelling of the feature name '
        #             'in the tied_embeddings field and '
        #             'be sure its type is not binary, numerical or timeseries.'.format(
        #                 self.tied_embeddings,
        #                 output_feature['name'],
        #                 self.tied_embeddings
        #             )
        #         )


        # if self.is_timeseries:
        #     vocab_size = 1
        # else:
        #     vocab_size = self.num_classes
        #
        # if not self.regularize:
        #     regularizer = None
        #
        # predictions_sequence, predictions_sequence_scores, \
        # predictions_sequence_length_with_eos, \
        # targets_sequence_length_with_eos, eval_logits, train_logits, \
        # class_weights, class_biases = recurrent_decoder(
        #     hidden,
        #     targets,
        #     output_feature['max_sequence_length'],
        #     vocab_size,
        #     cell_type=self.cell_type,
        #     state_size=self.state_size,
        #     embedding_size=self.embedding_size,
        #     beam_width=self.beam_width,
        #     num_layers=self.num_layers,
        #     attention_mechanism=self.attention_mechanism,
        #     is_timeseries=self.is_timeseries,
        #     embeddings=tied_embeddings_tensor,
        #     initializer=self.initializer,
        #     regularizer=regularizer
        # )
        #
        # probabilities_target_sequence = tf.nn.softmax(eval_logits)
        #
        # return predictions_sequence, predictions_sequence_scores, \
        #        predictions_sequence_length_with_eos, \
        #        probabilities_target_sequence, targets_sequence_length_with_eos, \
        #        eval_logits, train_logits, class_weights, class_biases


class SequenceTaggerDecoder(Layer):
    def __init__(
            self,
            num_classes,
            use_bias=True,
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            attention=False,
            is_timeseries=False,
            **kwargs
    ):
        super(SequenceTaggerDecoder, self).__init__()
        self.attention = attention

        if is_timeseries:
            num_classes = 1

        self.decoder_layer = Dense(
            units=num_classes,
            use_bias=use_bias,
            kernel_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer
        )

    def call(
            self,
            inputs,
            training=None,
            mask=None
    ):
        if len(inputs.shape) != 3:
            raise ValueError(
                'Decoder inputs rank is {}, but should be 3 [batch x sequence x hidden] '
                'when using a tagger sequential decoder. '
                'Consider setting reduce_output to null / None if a sequential encoder / combiner is used.'.format(
                    len(inputs.shape)))

        # hidden shape [batch_size, sequence_length, hidden_size]
        logits = self.decoder_layer(inputs)

        # TODO tf2 add feed forward attention

        # logits shape [batch_size, sequence_length, vocab_size]
        return logits


