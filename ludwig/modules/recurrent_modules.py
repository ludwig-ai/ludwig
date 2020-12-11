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
import inspect
import logging

from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Layer, SimpleRNN
from tensorflow.keras.layers import SimpleRNNCell, GRUCell, LSTMCell
from tensorflow.keras.layers import StackedRNNCells, AbstractRNNCell

from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)

rnn_layers_registry = {
    'rnn': SimpleRNN,
    'gru': GRU,
    'lstm': LSTM,
}

rnncell_layers_registry = {
    'rnn': SimpleRNNCell,
    'gru': GRUCell,
    'lstm': LSTMCell
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
        self.supports_masking = True

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

        for layer in self.layers:
            logger.debug('   {}'.format(layer.name))

    def call(self, inputs, training=None, mask=None):
        hidden = inputs
        final_state = None
        for layer in self.layers:
            outputs = layer(hidden, training=training, mask=mask)
            hidden = outputs[0]
            final_state = outputs[1:]
        if final_state:
            if len(final_state) == 1:
                final_state = final_state[0]
        return hidden, final_state

# todo: clean-up following
# this represents one approach, comments are here as reference during
# development
# class LudwigStackedRNNCells(StackedRNNCells):
#
#     # this was intended to fix an issue with TFA, however I'm thinking
#     # this is not the correct fix.
#     @property
#     def state_size(self):
#         state_size_tuple = tuple(c.state_size for c in
#                                  (self.cells[
#                                   ::-1] if self.reverse_state_order else self.cells))
#         if len(state_size_tuple) == 1:
#             return state_size_tuple
#         else:
#             return (state_size_tuple[0],)
#
#
# class RecurrentCellStack(StackedRNNCells):
#     def __init__(
#             self,
#             state_size=256,
#             cell_type='rnn',
#             num_layers=1,
#             **kwargs
#     ):
#         # super(RecurrentCellStack, self).__init__()
#
#         self._state_size = state_size
#         self.cell_type = cell_type
#         self.num_layers = num_layers
#
#         rnn_cell = get_from_registry(cell_type, rnncell_layers_registry)
#         rnn_cells = [rnn_cell(state_size) for _ in range(num_layers)]
#         super().__init__(rnn_cells)
#
#     # def call(self, inputs, states, constants=None, training=None, **kwargs):
#     #     return self(inputs, states, constants=constants,
#     #                          training=training,  **kwargs)
#
#     def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
#         state = []
#         for c in self.cells:
#             state.append(
#                 c.get_initial_state(
#                     inputs=inputs,
#                     batch_size=batch_size,
#                     dtype=dtype
#                 )
#             )
#         return state
#
#     # @property
#     # def output_size(self):
#     #     return self.stacked_rnn_cells.cells[-1].output_size
#
#     # @property
#     # def state_size(self):
#     #     cells = self.stacked_rnn_cells
#     #     state_size_tuple = tuple(c.state_size for c in
#     #                              (cells.cells[
#     #                               ::-1] if cells.reverse_state_order else cells.cells))
#     #     if len(state_size_tuple) == 1:
#     #         return state_size_tuple
#     #     else:
#     #         return (state_size_tuple[0],)
#
#     # def call(self, inputs, state, training=None):
#     #     return self.stacked_rnn_cells(inputs, state, training=training)
#
#     def get_config(self):
#         base_configs = super().get_config()
#         return {
#             **base_configs,
#             'state_size': self._state_size,
#             'cell_type': self.cell_type,
#             'num_layers': self.num_layers
#         }
