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
import tensorflow_addons as tfa
from tensorflow.keras.layers import GRUCell, SimpleRNNCell, LSTMCell
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras.layers import average
from tensorflow_addons.seq2seq import AttentionWrapper
from tensorflow_addons.seq2seq import BahdanauAttention
from tensorflow_addons.seq2seq import LuongAttention

from ludwig.constants import *
from ludwig.models.modules.reduction_modules import reduce_sequence
from ludwig.utils.misc import get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D, sequence_length_2D

# todo tf2 clean up
# from ludwig.models.modules.attention_modules import \
#     feed_forward_memory_attention
# from ludwig.models.modules.initializer_modules import get_initializer
# from ludwig.models.modules.recurrent_modules import recurrent_decoder

logger = logging.getLogger(__name__)

rnn_layers_registry = {
    'rnn': SimpleRNNCell,
    'gru': GRUCell,
    'lstm': LSTMCell
}


class SequenceGeneratorDecoder(Layer):
    def __init__(
            self,
            num_classes,
            cell_type='rnn',
            state_size=256,
            embedding_size=64,
            beam_width=1,
            num_layers=1,
            attention=None,
            tied_embeddings=None,
            initializer=None,
            regularize=True,
            is_timeseries=False,
            max_sequence_length=0,
            use_bias=True,
            weights_initializer='glorot_uniform',
            bias_initializer='zeros',
            weights_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            reduce_input='sum',
            **kwargs
    ):
        super(SequenceGeneratorDecoder, self).__init__()

        self.cell_type = cell_type
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.beam_width = beam_width
        self.num_layers = num_layers
        self.attention = attention
        self.attention_mechanism = None
        self.tied_embeddings = tied_embeddings
        self.initializer = initializer
        self.regularize = regularize
        self.is_timeseries = is_timeseries
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.state_size = state_size
        self.attention_mechanism = None

        self.reduce_input = reduce_input

        if is_timeseries:
            self.vocab_size = 1
        else:
            self.vocab_size = self.num_classes

        self.project = Dense(state_size)

        self.decoder_embedding = Embedding(
            input_dim=self.num_classes + 1,  # account for GO_SYMBOL
            output_dim=embedding_size)
        self.dense_layer = Dense(num_classes)
        self.decoder_rnncell = \
            get_from_registry(cell_type, rnn_layers_registry)(state_size)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        print('setting up attention for', attention)
        if attention is not None:
            if attention == 'luong':
                self.attention_mechanism = LuongAttention(units=state_size)
            elif attention == 'bahdanau':
                self.attention_mechanism = BahdanauAttention(units=state_size)

            self.decoder_rnncell = AttentionWrapper(self.decoder_rnncell,
                                                    self.attention_mechanism,
                                                    attention_layer_size=state_size)

        # self.decoder = tfa.seq2seq.BasicDecoder(self.decoder_rnncell,  # todo tf2 code clean
        #                                         sampler=self.sampler,
        #                                         output_layer=self.dense_layer)

    def _logits_training(self, inputs, target, training=None):
        input = inputs['hidden']  # shape [batch_size, seq_size, state_size]
        encoder_end_state = self._prepare_decoder_input_state(inputs)

        logits = self.decoder_training(
            input,
            target=target,
            encoder_end_state=encoder_end_state
        )
        return logits  # shape = [b, s, c]

    def _prepare_decoder_input_state(self, inputs):

        if 'encoder_output_state' in inputs:
            decoder_input_state = inputs['encoder_output_state']
        else:
            hidden = inputs['hidden']
            if len(hidden.shape) == 3:  # encoder_output is a sequence
                # reduce_sequence returns a [b, h]
                decoder_input_state = reduce_sequence(
                    hidden,
                    self.reduce_input if self.reduce_input else 'sum'
                )
            elif len(hidden.shape) == 2:
                # this returns a [b, h]
                decoder_input_state = hidden
            else:
                raise ValueError("Only works for 1d or 2d encoder_output")

        # now we have to deal with the fact that the state needs to be a list
        # in case of lstm or a tensor otherwise
        if (self.cell_type == 'lstm' and
                isinstance(decoder_input_state, list)):
            if len(decoder_input_state) == 2:
                # this maybe a unidirectionsl lstm or a bidirectional gru / rnn
                # there is no way to tell
                # If it is a unidirectional lstm, pass will work fine
                # if it is bidirectional gru / rnn, the output of one of
                # the directions will be treated as the inital c of the lstm
                # which is weird and may lead to poor performance
                # todo try to find a way to distinguish among these two cases
                pass
            elif len(decoder_input_state) == 4:
                # the encoder was a bidirectional lstm
                # a good strategy is to average the 2 h and the 2 c vectors
                decoder_input_state = [
                    average(
                        [decoder_input_state[0], decoder_input_state[2]]
                    ),
                    average(
                        [decoder_input_state[1], decoder_input_state[3]]
                    )
                ]
            else:
                # no idea how lists of length different than 2 or 4
                # might have been originated, we can either rise an ValueError
                # or deal with it averaging everything
                # raise ValueError(
                #     "encoder_output_state has length different than 2 or 4. "
                #     "Please doublecheck your encoder"
                # )
                average_state = average(decoder_input_state)
                decoder_input_state = [average_state, average_state]

        elif (self.cell_type == 'lstm' and
              not isinstance(decoder_input_state, list)):
            decoder_input_state = [decoder_input_state, decoder_input_state]

        elif (self.cell_type != 'lstm' and
              isinstance(decoder_input_state, list)):
            # here we have a couple options,
            # either reuse part of the input encoder state,
            # or just use its output
            if len(decoder_input_state) == 2:
                # using h and ignoring c
                decoder_input_state = decoder_input_state[0]
            elif len(decoder_input_state) == 4:
                # using average of hs and ignoring cs
                decoder_input_state + average(
                    [decoder_input_state[0], decoder_input_state[2]]
                )
            else:
                # no idea how lists of length different than 2 or 4
                # might have been originated, we can either rise an ValueError
                # or deal with it averaging everything
                # raise ValueError(
                #     "encoder_output_state has length different than 2 or 4. "
                #     "Please doublecheck your encoder"
                # )
                decoder_input_state = average(decoder_input_state)
            # this returns a [b, h]
            # decoder_input_state = reduce_sequence(eo, self.reduce_input)

        elif (self.cell_type != 'lstm' and
              not isinstance(decoder_input_state, list)):
            # do nothing, we are good
            pass

        # at this point decoder_input_state is either a [b,h]
        # or a list([b,h], [b,h]) if the decoder cell is an lstm
        # but h may not be the same as the decoder state size,
        # so we may need to project
        if isinstance(decoder_input_state, list):
            for i in range(len(decoder_input_state)):
                if (decoder_input_state[i].shape[1] !=
                        self.state_size):
                    decoder_input_state[i] = self.project(
                        decoder_input_state[i]
                    )
        else:
            if decoder_input_state.shape[1] != self.state_size:
                decoder_input_state = self.project(
                    decoder_input_state
                )

        # make sure we are passing back the state tensors in a list
        if not isinstance(decoder_input_state, list):
            decoder_input_state = [decoder_input_state]

        return decoder_input_state

    def build_decoder_initial_state(self, batch_size, encoder_state, dtype):
        decoder_initial_state = self.decoder_rnncell.get_initial_state(
            batch_size=batch_size,
            dtype=dtype)

        # handle situation where encoder and decoder are different cell_types
        # and to account for inconsistent wrapping for encoder state w/in lists
        if self.cell_type == 'lstm' and not isinstance(encoder_state, list):
            encoder_state = [encoder_state, encoder_state]
        elif self.cell_type != 'lstm' and isinstance(encoder_state, list):
            encoder_state = encoder_state[0]

        if self.attention_mechanism is not None:
            decoder_initial_state = decoder_initial_state.clone(
                    cell_state=encoder_state)
        else:
            if not isinstance(encoder_state, list):
                decoder_initial_state = [encoder_state]
            else:
                decoder_initial_state = encoder_state

        return decoder_initial_state

    def decoder_training(
            self,
            encoder_output,
            target=None,
            encoder_end_state=None
    ):
        # ================ Setup ================
        GO_SYMBOL = self.vocab_size
        END_SYMBOL = 0
        batch_size = encoder_output.shape[0]
        encoder_sequence_length = sequence_length_3D(encoder_output)

        # Prepare target for decoding
        target_sequence_length = sequence_length_2D(target)
        start_tokens = tf.tile([GO_SYMBOL], [batch_size])
        end_tokens = tf.tile([END_SYMBOL], [batch_size])
        if self.is_timeseries:
            start_tokens = tf.cast(start_tokens, tf.float32)
            end_tokens = tf.cast(end_tokens, tf.float32)
        targets_with_go_and_eos = tf.concat([
            tf.expand_dims(start_tokens, 1),
            target,  # todo tf2: right now cast to tf.int32, fails if tf.int64
            tf.expand_dims(end_tokens, 1)], 1)
        target_sequence_length_with_eos = target_sequence_length + 1

        # Decoder Embeddings
        decoder_emb_inp = self.decoder_embedding(targets_with_go_and_eos)

        # Setting up decoder memory from encoder output
        if self.attention_mechanism is not None:
            self.attention_mechanism.setup_memory(
                encoder_output,
                memory_sequence_length=encoder_sequence_length
            )

        decoder_initial_state = self.build_decoder_initial_state(
            batch_size,
            encoder_state=encoder_end_state,
            dtype=tf.float32
        )

        decoder = tfa.seq2seq.BasicDecoder(
            self.decoder_rnncell,
            sampler=self.sampler,
            output_layer=self.dense_layer
        )

        # BasicDecoderOutput
        outputs, final_state, generated_sequence_lengths = decoder(
            decoder_emb_inp,
            initial_state=decoder_initial_state,
            sequence_length=target_sequence_length_with_eos
        )

        logits = outputs.rnn_output
        mask = tf.sequence_mask(
            generated_sequence_lengths,
            maxlen=logits.shape[1],
            dtype=tf.float32
        )
        logits = logits * mask[:, :, tf.newaxis]
        return logits  # , outputs, final_state, generated_sequence_lengths


    def decoder_beam_search(
        self,
        encoder_output,
        encoder_end_state=None,
        training=None
    ):

        # ================ Setup ================
        GO_SYMBOL = self.vocab_size
        END_SYMBOL = 0
        batch_size = encoder_output.shape[0]
        encoder_sequence_length = sequence_length_3D(encoder_output)

        # ================ predictions =================
        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        decoder_input = tf.expand_dims(
            [GO_SYMBOL] * batch_size, 1)
        start_tokens = tf.fill([batch_size], GO_SYMBOL)
        end_token = END_SYMBOL

        decoder_emb_inp = self.decoder_embedding(decoder_input)

        # code sequence based on example found here
        # https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/BeamSearchDecoder
        tiled_encoder_output = tfa.seq2seq.tile_batch(
            encoder_output,
            multiplier=self.beam_width
        )

        tiled_encoder_end_state = tfa.seq2seq.tile_batch(
            encoder_end_state,
            multiplier=self.beam_width
        )

        tiled_encoder_sequence_length = tfa.seq2seq.tile_batch(
            encoder_sequence_length,
            multiplier=self.beam_width
        )

        if self.attention_mechanism is not None:
            self.attention_mechanism.setup_memory(
                tiled_encoder_output,
                memory_sequence_length=tiled_encoder_sequence_length
            )

        decoder_initial_state = self.build_decoder_initial_state(
            batch_size * self.beam_width,
            encoder_state=tiled_encoder_end_state,
            dtype=tf.float32
        )

        decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
            cell=self.decoder_rnncell,
            beam_width=self.beam_width,
            output_layer=self.dense_layer)
        # ================generate logits ==================
        maximum_iterations = self.max_sequence_length

        # initialize inference decoder
        decoder_embedding_matrix = self.decoder_embedding.variables[0]
        (
            first_finished,
            first_inputs,
            first_state
        ) = decoder.initialize(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            # following construct required to work around inconsistent handling
            # of encoder_end_state by tfa
            initial_state=decoder_initial_state \
                if len(decoder_initial_state) != 1 else decoder_initial_state[
                0]
        )

        inputs = first_inputs
        state = first_state

        # create empty logits tensor
        logits = tf.convert_to_tensor(
            np.array([]).reshape([batch_size, 0, self.num_classes]),
            dtype=tf.float32
        )
        # create empty predictions tensor
        predictions = tf.convert_to_tensor(
            np.array([]).reshape([batch_size, 0]),
            dtype=tf.int32  # todo tf2 need to change to tf.int64
        )

        # beam search
        for j in range(maximum_iterations):
            outputs, next_state, next_inputs, finished = decoder.step(
                j, inputs, state, training=training)
            inputs = next_inputs
            state = next_state
            # logtis don't work, temporary workaround
            one_logit = tf.zeros([batch_size, 1, self.num_classes])
            logits = tf.concat([logits, one_logit], axis=1)
            one_predicted_token = tf.expand_dims(outputs.predicted_ids[:, 0],
                                                 axis=1)
            predictions = tf.concat([predictions, one_predicted_token], axis=1)

        # todo we should first run all the iterations and only at the end
        #  collect logits and predictions

        # todo solve cases when predictions become 0 and then return
        #  to be a number, which confuses the last_predictions later

        return logits, predictions


    def decoder_inference(
        self,
        encoder_output,
        encoder_end_state=None,
        training=None
    ):

        # ================ Setup ================
        GO_SYMBOL = self.vocab_size
        END_SYMBOL = 0
        batch_size = encoder_output.shape[0]
        encoder_sequence_length = sequence_length_3D(encoder_output)

        # ================ predictions =================
        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        decoder_input = tf.expand_dims(
            [GO_SYMBOL] * batch_size, 1)
        start_tokens = tf.fill([batch_size], GO_SYMBOL)
        end_token = END_SYMBOL

        decoder_emb_inp = self.decoder_embedding(decoder_input)

        if self.attention_mechanism is not None:
            self.attention_mechanism.setup_memory(
                encoder_output,
                memory_sequence_length=encoder_sequence_length
            )

        decoder_initial_state = self.build_decoder_initial_state(
            batch_size,
            encoder_state=encoder_end_state,
            dtype=tf.float32
        )

        decoder = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_rnncell,
            sampler=greedy_sampler,
            output_layer=self.dense_layer
        )

        # ================generate logits ==================
        maximum_iterations = self.max_sequence_length

        # initialize inference decoder
        decoder_embedding_matrix = self.decoder_embedding.variables[0]
        (
            first_finished,
            first_inputs,
            first_state
        ) = decoder.initialize(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state
        )

        inputs = first_inputs
        state = first_state

        # create empty logits tensor
        logits = tf.convert_to_tensor(
            np.array([]).reshape([batch_size, 0, self.num_classes]),
            dtype=tf.float32
        )
        # create empty predictions tensor
        predictions = tf.convert_to_tensor(
            np.array([]).reshape([batch_size, 0]),
            dtype=tf.int32  # todo tf2 need to change to tf.int64
        )

        # build up logits
        for j in range(maximum_iterations):
            outputs, next_state, next_inputs, finished = decoder.step(
                j, inputs, state, training=training)
            inputs = next_inputs
            state = next_state
            one_logit = tf.expand_dims(outputs.rnn_output, axis=1)
            logits = tf.concat([logits, one_logit], axis=1)
            one_prediction = tf.expand_dims(outputs.sample_id, axis=1)
            predictions = tf.concat([predictions, one_prediction], axis=1)

        # todo solve cases when predictions become 0 and then return
        #  to be a number, which confuses the last_predictions later

        return logits, predictions

    # this should be used only for decoder inference
    def call(self, inputs, training=None, mask=None):
        # shape [batch_size, seq_size, state_size]
        encoder_output = inputs[LOGITS]['hidden']
        # form dependent on cell_type
        # lstm: list([batch_size, state_size], [batch_size, state_size])
        # rnn, gru: [batch_size, state_size]
        encoder_output_state = self._prepare_decoder_input_state(
            inputs[LOGITS]
        )

        if self.beam_width > 1:
            logits, predictions = self.decoder_beam_search(
                encoder_output,
                encoder_end_state=encoder_output_state,
                training=training
            )
        else:
            logits, predictions = self.decoder_inference(
                encoder_output,
                encoder_end_state=encoder_output_state,
                training=training
            )

        return logits, predictions

    # def call(
    #         self,
    #         inputs,
    #         training=None,
    #         mask=None,
    #         target=None
    # ):
    # input = inputs['hidden']
    # try:
        #     encoder_output_state = inputs['encoder_output_state']
        # except KeyError:
        #     encoder_output_state = None
        #
        # todo tf2 need to move this to sequence output feature class
        # if len(input.shape) != 3 and self.attention_mechanism is not None:
        #     raise ValueError(
        #         'Encoder outputs rank is {}, but should be 3 [batch x sequence x hidden] '
        #         'when attention mechanism is {}. '
        #         'If you are using a sequential encoder or combiner consider setting reduce_output to None '
        #         'and flatten to False if those parameters apply.'
        #         'Also make sure theat reduce_input of output feature is None,'.format(
        #             len(input.shape), self.attention_name))
        # if len(input.shape) != 2 and self.attention_mechanism is None:
        #     raise ValueError(
        #         'Encoder outputs rank is {}, but should be 2 [batch x hidden] '
        #         'when attention mechanism is {}. '
        #         'Consider setting reduce_input of output feature to a value different from None.'.format(
        #             len(input.shape), self.attention_name))
        #
        # batch_size = input.shape[0]
        # print(">>>>>>batch_size", batch_size)
        #
        # # Assume we have a final state
        # encoder_end_state = encoder_output_state
        #
        # # in case we don't have a final state set to default value
        # if self.cell_type in 'lstm' and encoder_end_state is None:
        #     encoder_end_state = [
        #         tf.zeros([batch_size, self.rnn_units], tf.float32),
        #         tf.zeros([batch_size, self.rnn_units], tf.float32)
        #     ]
        # elif self.cell_type in {'rnn', 'gru'} and encoder_end_state is None:
        #     encoder_end_state = tf.zeros([batch_size, self.rnn_units], tf.float32)



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

        self.projection_layer = Dense(
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
        hidden = inputs[HIDDEN]
        if len(hidden.shape) != 3:
            raise ValueError(
                'Decoder inputs rank is {}, but should be 3 [batch x sequence x hidden] '
                'when using a tagger sequential decoder. '
                'Consider setting reduce_output to null / None if a sequential encoder / combiner is used.'.format(
                    len(hidden.shape)))

        # TODO tf2 add feed forward attention

        # hidden shape [batch_size, sequence_length, hidden_size]
        logits = self.projection_layer(hidden)

        return logits, None  # logits shape [batch_size, sequence_length, num_classes]

    def _logits_training(self, inputs, **kwargs):
        return self(inputs)
