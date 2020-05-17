# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
import collections
import inspect
import logging

import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
from tensorflow.compat.v1.nn.rnn_cell import MultiRNNCell, LSTMStateTuple
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Bidirectional, Layer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

# from ludwig.models.modules.fully_connected_modules import fc_layer
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.reduction_modules import reduce_sequence
from ludwig.utils.misc import get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D, sequence_length_2D

logger = logging.getLogger(__name__)

rnn_layers_registry = {
    'rnn': SimpleRNN,
    'gru': GRU,
    'lstm': LSTM,
}


class RecurrentStack(Layer):
    def __init__(
            self,
            state_size=256,
            cell_type='rnn',
            num_layers=1,
            bidirectional=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            unit_forget_bias=True,
            weights_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            weights_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # kernel_constraint=kernel_constraint,
            # recurrent_constraint=recurrent_constraint,
            # bias_constraint=bias_constraint,
            dropout=0.0,
            recurrent_dropout=0.0,
            **kwargs
    ):
        super(RecurrentStack, self).__init__()

        rnn_layer_class = get_from_registry(cell_type, rnn_layers_registry)
        self.layers = []

        rnn_params = {
            'units': state_size,
            'activation': activation,
            'recurrent_activation': recurrent_activation,
            'use_bias': use_bias,
            'kernel_initializer': weights_initializer,
            'recurrent_initializer': recurrent_initializer,
            'bias_initializer': bias_initializer,
            'unit_forget_bias': unit_forget_bias,
            'kernel_regularizer': weights_regularizer,
            'recurrent_regularizer': recurrent_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            # 'kernel_constraint': weights_constraint,
            # 'recurrent_constraint': recurrent_constraint,
            # 'bias_constraint': bias_constraint,
            'dropout': dropout,
            'recurrent_dropout': recurrent_dropout,
            'return_sequences': True,
            'return_state': True,
        }
        signature = inspect.signature(rnn_layer_class.__init__)
        valid_args = set(signature.parameters.keys())
        rnn_params = {k: v for k, v in rnn_params.items() if k in valid_args}

        for _ in range(num_layers):
            layer = rnn_layer_class(**rnn_params)

            if bidirectional:
                layer = Bidirectional(layer)

            self.layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        hidden = inputs
        final_state = None
        for layer in self.layers:
            outputs = layer(hidden, training=training)
            hidden = outputs[0]
            final_state = outputs[1:]
        if final_state:
            if len(final_state) == 1:
                final_state = final_state[0]
        return hidden, final_state

def get_cell_fun(cell_type):
    if cell_type == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell  # todo tf2: do we eventually need tf2.keras.layers.SimpleRNNCell
    elif cell_type == 'lstm':
        # allows for optional peephole connections and cell clipping
        cell_fn = tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'lstm_block':
        # Faster version of basic LSTM
        cell_fn = tfa.rnn.LSTMBlockCell
    elif cell_type == 'lstm_ln':
        cell_fn = tfa.rnn.LayerNormBasicLSTMCell
    elif cell_type == 'lstm_cudnn':
        cell_fn = tfa.cudnn_rnn.CudnnCompatibleLSTMCell
    elif cell_type == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif cell_type == 'gru_block':
        # Faster version of GRU (25% faster in my tests)
        cell_fn = tfa.rnn.GRUBlockCell
    elif cell_type == 'gru_cudnn':
        # Faster version of GRU (25% faster in my tests)
        cell_fn = tfa.cudnn_rnn.CudnnCompatibleGRUCell
    else:
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    return cell_fn


class Projection(tf.layers.Layer):
    def __init__(self, projection_weights, projection_biases, name=None,
                 **kwargs):
        super(Projection, self).__init__(name=name, **kwargs)
        self.projection_weights = projection_weights
        self.projection_biases = projection_biases

    def call(self, inputs, **kwargs):
        inputs_shape = inputs.shape.as_list()
        weights_shape = self.projection_weights.shape.as_list()
        assert inputs_shape[-1] == weights_shape[0]
        inputs = tf.reshape(inputs, [-1, inputs_shape[-1]])

        outputs = tf.matmul(inputs, self.projection_weights)
        if self.projection_biases is not None:
            outputs = tf.nn.bias_add(outputs, self.projection_biases)

        outputs_shape = inputs_shape
        outputs_shape[0] = -1  # batch_size
        outputs_shape[-1] = weights_shape[1]
        outputs = tf.reshape(outputs, outputs_shape)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = input_shape
        output_shape[-1] = self.projection_biases.shape.as_list()[0]
        # output_shape = [input_shape[0], self.projection_biases.shape.as_list()[0]]
        return tensor_shape.TensorShape(output_shape)


class BasicDecoderOutput(
    collections.namedtuple('BasicDecoderOutput',
                           ('rnn_output', 'sample_id', 'projection_input'))):
    pass


class BasicDecoder(tfa.seq2seq.BasicDecoder):
    def _projection_input_size(self):
        return self.cell.output_size

    @property
    def output_size(self):
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=self.sampler.sample_ids_shape,
            projection_input=self._projection_input_size())

    @property
    def output_dtype(self):
        dtype = self._cell_dtype
        return BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self.sampler.sample_ids_dtype,
            nest.map_structure(lambda _: dtype, self._projection_input_size()))

    def step(self, time, inputs, state, name=None):
        with ops.name_scope(name, 'BasicDecoderStep', (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            projection_inputs = cell_outputs  # get projection_inputs to compute sampled_softmax_cross_entropy_loss
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids,
                                     projection_inputs)
        return (outputs, next_state, next_inputs, finished)


def recurrent_decoder(encoder_outputs, targets, max_sequence_length, vocab_size,
                      cell_type='rnn', state_size=256, embedding_size=50,
                      num_layers=1,
                      attention_mechanism=None, beam_width=1, projection=True,
                      tied_target_embeddings=True, embeddings=None,
                      initializer=None, regularizer=None):
    with tf.variable_scope('rnn_decoder', reuse=tf.AUTO_REUSE,
                           regularizer=regularizer):

        # ================ Setup ================
        GO_SYMBOL = vocab_size
        END_SYMBOL = 0
        batch_size = tf.shape(encoder_outputs)[0]

        # ================ Projection ================
        # Project the encoder outputs to the size of the decoder state
        encoder_outputs_size = encoder_outputs.shape[-1]
        if projection and encoder_outputs_size != state_size:
            with tf.variable_scope('projection'):
                encoder_output_rank = len(encoder_outputs.shape)
                if encoder_output_rank > 2:
                    sequence_length = tf.shape(encoder_outputs)[1]
                    encoder_outputs = tf.reshape(encoder_outputs,
                                                 [-1, encoder_outputs_size])
                    # encoder_outputs = fc_layer(encoder_outputs,
                    #                           encoder_outputs.shape[-1],
                    #                           state_size,
                    #                           activation=None,
                    #                           initializer=initializer)
                    encoder_outputs = tf.reshape(encoder_outputs,
                                                 [-1, sequence_length,
                                                  state_size])
                # else:
                #    encoder_outputs = fc_layer(encoder_outputs,
                #                               encoder_outputs.shape[-1],
                #                               state_size,
                #                               activation=None,
                #                               initializer=initializer)

        # ================ Targets sequence ================
        # Calculate the length of inputs and the batch size
        with tf.variable_scope('sequence'):
            targets_sequence_length = sequence_length_2D(targets)
            start_tokens = tf.tile([GO_SYMBOL], [batch_size])
            end_tokens = tf.tile([END_SYMBOL], [batch_size])
            targets_with_go_and_eos = tf.concat([
                tf.expand_dims(start_tokens, 1),
                targets,
                tf.expand_dims(end_tokens, 1)], 1)
            logger.debug(
                '  targets_with_go: {0}'.format(targets_with_go_and_eos))
            targets_sequence_length_with_eos = targets_sequence_length + 1  # the EOS symbol is 0 so it's not increasing the real length of the sequence

        # ================ Embeddings ================
        with tf.variable_scope('embedding'):
            if embeddings is not None:
                embedding_size = embeddings.shape.as_list()[-1]
                if tied_target_embeddings:
                    state_size = embedding_size
            elif tied_target_embeddings:
                embedding_size = state_size

            if embeddings is not None:
                embedding_go = tf.get_variable('embedding_GO',
                                               initializer=tf.random_uniform(
                                                   [1, embedding_size],
                                                   -1.0, 1.0))
                targets_embeddings = tf.concat([embeddings, embedding_go],
                                               axis=0)
            else:
                initializer_obj = get_initializer(initializer)
                targets_embeddings = tf.get_variable(
                    'embeddings',
                    initializer=initializer_obj(
                        [vocab_size + 1, embedding_size]),
                    regularizer=regularizer
                )
            logger.debug(
                '  targets_embeddings: {0}'.format(targets_embeddings))

            targets_embedded = tf.nn.embedding_lookup(targets_embeddings,
                                                      targets_with_go_and_eos,
                                                      name='decoder_input_embeddings')
        logger.debug('  targets_embedded: {0}'.format(targets_embedded))

        # ================ Class prediction ================
        if tied_target_embeddings:
            class_weights = tf.transpose(targets_embeddings)
        else:
            initializer_obj = get_initializer(initializer)
            class_weights = tf.get_variable(
                'class_weights',
                initializer=initializer_obj([state_size, vocab_size + 1]),
                regularizer=regularizer
            )
        logger.debug('  class_weights: {0}'.format(class_weights))
        class_biases = tf.get_variable('class_biases', [vocab_size + 1])
        logger.debug('  class_biases: {0}'.format(class_biases))
        projection_layer = Projection(class_weights, class_biases)

        # ================ RNN ================
        initial_state = encoder_outputs
        with tf.variable_scope('rnn_cells') as vs:
            # Cell
            cell_fun = get_cell_fun(cell_type)

            if num_layers == 1:
                cell = cell_fun(state_size)
                if cell_type.startswith('lstm'):
                    initial_state = LSTMStateTuple(c=initial_state,
                                                   h=initial_state)
            elif num_layers > 1:
                cell = MultiRNNCell(
                    [cell_fun(state_size) for _ in range(num_layers)],
                    state_is_tuple=True)
                if cell_type.startswith('lstm'):
                    initial_state = LSTMStateTuple(c=initial_state,
                                                   h=initial_state)
                initial_state = tuple([initial_state] * num_layers)
            else:
                raise ValueError('num_layers in recurrent decoser: {}. '
                                 'Number of layers in a recurrenct decoder cannot be <= 0'.format(
                    num_layers))

            # Attention
            if attention_mechanism is not None:
                if attention_mechanism == 'bahdanau':
                    attention_mechanism = tfa.seq2seq.BahdanauAttention(
                        num_units=state_size, memory=encoder_outputs,
                        memory_sequence_length=sequence_length_3D(
                            encoder_outputs))
                elif attention_mechanism == 'luong':
                    attention_mechanism = tfa.seq2seq.LuongAttention(
                        num_units=state_size, memory=encoder_outputs,
                        memory_sequence_length=sequence_length_3D(
                            encoder_outputs))
                else:
                    raise ValueError(
                        'Attention mechanism {} not supported'.format(
                            attention_mechanism))
                cell = tfa.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=state_size)
                initial_state = cell.zero_state(
                    dtype=tf.float32,
                    batch_size=batch_size)
                initial_state = initial_state.clone(
                    cell_state=reduce_sequence(encoder_outputs, 'last'))

            for v in tf.global_variables():
                if v.name.startswith(vs.name):
                    logger.debug('  {}: {}'.format(v.name, v))

        # ================ Decoding ================
        def decode(initial_state, cell, sampler, beam_width=1,
                   projection_layer=None, inputs=None):
            # The decoder itself
            if beam_width > 1:
                # Tile inputs for beam search decoder
                beam_initial_state = tfa.seq2seq.tile_batch(
                    initial_state, beam_width)
                decoder = tfa.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=targets_embeddings,
                    start_tokens=start_tokens,
                    end_token=END_SYMBOL,
                    initial_state=beam_initial_state,
                    beam_width=beam_width,
                    output_layer=projection_layer)
            else:
                decoder = BasicDecoder(
                    cell=cell, sampler=sampler,
                    output_layer=projection_layer)
                # todo tf2: remove obsolete code #initial_state=initial_state,

                # todo tf2: need to figure out 'inputs' to next function
                decoder.initialize(inputs, initial_state=initial_state)

            # The decoding operation
            outputs = tfa.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False if beam_width > 1 else True,
                maximum_iterations=max_sequence_length,
                decoder_init_input=inputs,
                decoder_init_kwargs={'initial_state': initial_state}
            )

            return outputs

        # ================ Decoding helpers ================

        train_sampler = tfa.seq2seq.sampler.TrainingSampler()
        train_sampler.initialize(targets_embedded,
                                 sequence_length=targets_sequence_length_with_eos)
        # todo tf2: cleanout obsolete code
        # train_helper = tfa.seq2seq.sampler.TrainingSampler(
        #     inputs=targets_embedded,
        #     sequence_length=targets_sequence_length_with_eos)

        # # todo tf2: test code
        # initial_state = cell.get_initial_state(
        #     batch_size=batch_size, dtype=tf.float32
        # )

        final_outputs_train, final_state_train, final_sequence_lengths_train = decode(
            initial_state,
            cell,
            train_sampler,  # todo: tf2 to be removed #train_helper,
            projection_layer=projection_layer,
            inputs=encoder_outputs
        )
        eval_logits = final_outputs_train.rnn_output
        train_logits = final_outputs_train.projection_input
        # train_predictions = final_outputs_train.sample_id

        pred_helper = tfa.seq2seq.sampler.GreedyEmbeddingSampler(
            embedding=targets_embeddings,
            start_tokens=start_tokens,
            end_token=END_SYMBOL)
        final_outputs_pred, final_state_pred, final_sequence_lengths_pred = decode(
            initial_state,
            cell,
            pred_helper,
            beam_width,
            projection_layer=projection_layer)

        if beam_width > 1:
            predictions_sequence = final_outputs_pred.beam_search_decoder_output.predicted_ids[
                                   :, :, 0]
            # final_outputs_pred..predicted_ids[:,:,0] would work too, but it contains -1s for padding
            predictions_sequence_scores = final_outputs_pred.beam_search_decoder_output.scores[
                                          :, :, 0]
            predictions_sequence_length_with_eos = final_sequence_lengths_pred[
                                                   :, 0]
        else:
            predictions_sequence = final_outputs_pred.sample_id
            predictions_sequence_scores = final_outputs_pred.rnn_output
            predictions_sequence_length_with_eos = final_sequence_lengths_pred

    logger.debug('  train_logits: {0}'.format(train_logits))
    logger.debug('  eval_logits: {0}'.format(eval_logits))
    logger.debug('  predictions_sequence: {0}'.format(predictions_sequence))
    logger.debug('  predictions_sequence_scores: {0}'.format(
        predictions_sequence_scores))

    return (
        predictions_sequence,
        predictions_sequence_scores,
        predictions_sequence_length_with_eos,
        targets_sequence_length_with_eos,
        eval_logits,
        train_logits,
        class_weights,
        class_biases
    )
