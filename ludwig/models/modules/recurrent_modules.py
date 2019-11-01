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
import logging

import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, LSTMStateTuple
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

from ludwig.models.modules.fully_connected_modules import fc_layer
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.reduction_modules import reduce_sequence
from ludwig.utils.tf_utils import sequence_length_3D, sequence_length_2D


logger = logging.getLogger(__name__)


def get_cell_fun(cell_type):
    if cell_type == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif cell_type == 'lstm':
        # allows for optional peephole connections and cell clipping
        cell_fn = tf.nn.rnn_cell.LSTMCell
    elif cell_type == 'lstm_block':
        # Faster version of basic LSTM
        cell_fn = tf.contrib.rnn.LSTMBlockCell
    elif cell_type == 'lstm_ln':
        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
    elif cell_type == 'lstm_cudnn':
        cell_fn = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
    elif cell_type == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif cell_type == 'gru_block':
        # Faster version of GRU (25% faster in my tests)
        cell_fn = tf.contrib.rnn.GRUBlockCell
    elif cell_type == 'gru_cudnn':
        # Faster version of GRU (25% faster in my tests)
        cell_fn = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell
    else:
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    return cell_fn


class Projection(tf.compat.v1.layers.Layer):
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


class BasicDecoder(tf.contrib.seq2seq.BasicDecoder):
    def _projection_input_size(self):
        return self._cell.output_size

    @property
    def output_size(self):
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            projection_input=self._projection_input_size())

    @property
    def output_dtype(self):
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self._helper.sample_ids_dtype,
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


class TimeseriesTrainingHelper(tf.contrib.seq2seq.TrainingHelper):
    def sample(self, time, outputs, name=None, **unused_kwargs):
        with ops.name_scope(name, 'TrainingHelperSample', [time, outputs]):
            return tf.zeros(tf.shape(outputs)[:-1], dtype=dtypes.int32)


class RecurrentStack:
    def __init__(
            self,
            state_size=256,
            cell_type='rnn',
            num_layers=1,
            bidirectional=False,
            dropout=False,
            regularize=True,
            reduce_output='last',
            **kwargs
    ):
        self.state_size = state_size
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.regularize = regularize
        self.reduce_output = reduce_output

    def __call__(
            self,
            input_sequence,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        if not self.regularize:
            regularizer = None

        # Calculate the length of input_sequence and the batch size
        sequence_length = sequence_length_3D(input_sequence)

        # RNN cell
        cell_fn = get_cell_fun(self.cell_type)

        # initial state
        # init_state = tf.compat.v1.get_variable(
        #   'init_state',
        #   [1, state_size],
        #   initializer=tf.constant_initializer(0.0),
        # )
        # init_state = tf.tile(init_state, [batch_size, 1])

        # main RNN operation
        with tf.compat.v1.variable_scope('rnn_stack', reuse=tf.compat.v1.AUTO_REUSE,
                               regularizer=regularizer) as vs:
            if self.bidirectional:
                # forward direction cell
                fw_cell = lambda state_size: cell_fn(state_size)
                bw_cell = lambda state_size: cell_fn(state_size)
                fw_cells = [fw_cell(self.state_size) for _ in
                            range(self.num_layers)]
                bw_cells = [bw_cell(self.state_size) for _ in
                            range(self.num_layers)]
                rnn_outputs, final_state_fw, final_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=fw_cells,
                    cells_bw=bw_cells,
                    dtype=tf.float32,
                    sequence_length=sequence_length,
                    inputs=input_sequence
                )

            else:
                cell = lambda state_size: cell_fn(state_size)
                cells = MultiRNNCell(
                    [cell(self.state_size) for _ in range(self.num_layers)],
                    state_is_tuple=True)
                rnn_outputs, final_state = tf.nn.dynamic_rnn(
                    cells,
                    input_sequence,
                    sequence_length=sequence_length,
                    dtype=tf.float32)
                # initial_state=init_state)

            for v in tf.global_variables():
                if v.name.startswith(vs.name):
                    logger.debug('  {}: {}'.format(v.name, v))
            logger.debug('  rnn_outputs: {0}'.format(rnn_outputs))

            rnn_output = reduce_sequence(rnn_outputs, self.reduce_output)
            logger.debug('  reduced_rnn_output: {0}'.format(rnn_output))

        # dropout
        if self.dropout and dropout_rate is not None:
            rnn_output = tf.layers.dropout(
                rnn_output,
                rate=dropout_rate,
                training=is_training
            )
            logger.debug('  dropout_rnn: {0}'.format(rnn_output))

        return rnn_output, rnn_output.shape.as_list()[-1]


def recurrent_decoder(encoder_outputs, targets, max_sequence_length, vocab_size,
                      cell_type='rnn', state_size=256, embedding_size=50,
                      num_layers=1,
                      attention_mechanism=None, beam_width=1, projection=True,
                      tied_target_embeddings=True, embeddings=None,
                      initializer=None, regularizer=None,
                      is_timeseries=False):
    with tf.compat.v1.variable_scope('rnn_decoder', reuse=tf.compat.v1.AUTO_REUSE,
                           regularizer=regularizer):

        # ================ Setup ================
        if beam_width > 1 and is_timeseries:
            raise ValueError('Invalid beam_width: {}'.format(beam_width))

        GO_SYMBOL = vocab_size
        END_SYMBOL = 0
        batch_size = tf.shape(encoder_outputs)[0]

        # ================ Projection ================
        # Project the encoder outputs to the size of the decoder state
        encoder_outputs_size = encoder_outputs.shape[-1]
        if projection and encoder_outputs_size != state_size:
            with tf.compat.v1.variable_scope('projection'):
                encoder_output_rank = len(encoder_outputs.shape)
                if encoder_output_rank > 2:
                    sequence_length = tf.shape(encoder_outputs)[1]
                    encoder_outputs = tf.reshape(encoder_outputs,
                                                 [-1, encoder_outputs_size])
                    encoder_outputs = fc_layer(encoder_outputs,
                                               encoder_outputs.shape[-1],
                                               state_size,
                                               activation=None,
                                               initializer=initializer)
                    encoder_outputs = tf.reshape(encoder_outputs,
                                                 [-1, sequence_length,
                                                  state_size])
                else:
                    encoder_outputs = fc_layer(encoder_outputs,
                                               encoder_outputs.shape[-1],
                                               state_size,
                                               activation=None,
                                               initializer=initializer)

        # ================ Targets sequence ================
        # Calculate the length of inputs and the batch size
        with tf.compat.v1.variable_scope('sequence'):
            targets_sequence_length = sequence_length_2D(targets)
            start_tokens = tf.tile([GO_SYMBOL], [batch_size])
            end_tokens = tf.tile([END_SYMBOL], [batch_size])
            if is_timeseries:
                start_tokens = tf.cast(start_tokens, tf.float32)
                end_tokens = tf.cast(end_tokens, tf.float32)
            targets_with_go_and_eos = tf.concat([
                tf.expand_dims(start_tokens, 1),
                targets,
                tf.expand_dims(end_tokens, 1)], 1)
            logger.debug('  targets_with_go: {0}'.format(targets_with_go_and_eos))
            targets_sequence_length_with_eos = targets_sequence_length + 1  # the EOS symbol is 0 so it's not increasing the real length of the sequence

        # ================ Embeddings ================
        if is_timeseries:
            targets_embedded = tf.expand_dims(targets_with_go_and_eos, -1)
            targets_embeddings = None
        else:
            with tf.compat.v1.variable_scope('embedding'):
                if embeddings is not None:
                    embedding_size = embeddings.shape.as_list()[-1]
                    if tied_target_embeddings:
                        state_size = embedding_size
                elif tied_target_embeddings:
                    embedding_size = state_size

                if embeddings is not None:
                    embedding_go = tf.compat.v1.get_variable('embedding_GO',
                                                   initializer=tf.random_uniform(
                                                       [1, embedding_size],
                                                       -1.0, 1.0))
                    targets_embeddings = tf.concat([embeddings, embedding_go],
                                                   axis=0)
                else:
                    initializer_obj = get_initializer(initializer)
                    targets_embeddings = tf.compat.v1.get_variable(
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
            class_weights = tf.compat.v1.get_variable(
                'class_weights',
                initializer=initializer_obj([state_size, vocab_size + 1]),
                regularizer=regularizer
            )
        logger.debug('  class_weights: {0}'.format(class_weights))
        class_biases = tf.compat.v1.get_variable('class_biases', [vocab_size + 1])
        logger.debug('  class_biases: {0}'.format(class_biases))
        projection_layer = Projection(class_weights, class_biases)

        # ================ RNN ================
        initial_state = encoder_outputs
        with tf.compat.v1.variable_scope('rnn_cells') as vs:
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
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=state_size, memory=encoder_outputs,
                        memory_sequence_length=sequence_length_3D(
                            encoder_outputs))
                elif attention_mechanism == 'luong':
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                        num_units=state_size, memory=encoder_outputs,
                        memory_sequence_length=sequence_length_3D(
                            encoder_outputs))
                else:
                    raise ValueError(
                        'Attention mechanism {} not supported'.format(
                            attention_mechanism))
                cell = tf.contrib.seq2seq.AttentionWrapper(
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
        def decode(initial_state, cell, helper, beam_width=1,
                   projection_layer=None):
            # The decoder itself
            if beam_width > 1:
                # Tile inputs for beam search decoder
                beam_initial_state = tf.contrib.seq2seq.tile_batch(
                    initial_state, beam_width)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=targets_embeddings,
                    start_tokens=start_tokens,
                    end_token=END_SYMBOL,
                    initial_state=beam_initial_state,
                    beam_width=beam_width,
                    output_layer=projection_layer)
            else:
                decoder = BasicDecoder(
                    cell=cell, helper=helper,
                    initial_state=initial_state,
                    output_layer=projection_layer)

            # The decoding operation
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False if beam_width > 1 else True,
                maximum_iterations=max_sequence_length
            )

            return outputs

        # ================ Decoding helpers ================
        if is_timeseries:
            train_helper = TimeseriesTrainingHelper(
                inputs=targets_embedded,
                sequence_length=targets_sequence_length_with_eos)
            final_outputs_pred, final_state_pred, final_sequence_lengths_pred = decode(
                initial_state,
                cell,
                train_helper,
                projection_layer=projection_layer)
            eval_logits = final_outputs_pred.rnn_output
            train_logits = final_outputs_pred.projection_input
            predictions_sequence = tf.reshape(eval_logits, [batch_size, -1])
            predictions_sequence_length_with_eos = final_sequence_lengths_pred

        else:
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=targets_embedded,
                sequence_length=targets_sequence_length_with_eos)
            final_outputs_train, final_state_train, final_sequence_lengths_train = decode(
                initial_state,
                cell,
                train_helper,
                projection_layer=projection_layer)
            eval_logits = final_outputs_train.rnn_output
            train_logits = final_outputs_train.projection_input
            # train_predictions = final_outputs_train.sample_id

            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
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
