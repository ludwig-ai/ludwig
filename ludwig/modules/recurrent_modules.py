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
import collections

import torch
from torch.nn import RNN, GRU, LSTM
from ludwig.utils.torch_utils import LudwigModule

from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)

rnn_layers_registry = {
    'rnn': RNN,
    'gru': GRU,
    'lstm': LSTM,
}


class RecurrentStack(LudwigModule):
    def __init__(
            self,
            input_size=None,
            hidden_size=256,
            cell_type='rnn',
            sequence_size=None,
            num_layers=1,
            bidirectional=False,
            activation='tanh',
            nonlinearity='tanh',
            recurrent_activation='signmoid',
            use_bias=True,
            unit_forget_bias=True,
            weights_initializer='xavier_uniform',
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
        # todo: Need to account for all the options, e.g., num_layers, bidirectional,
        #       initializers, regularizers, etc.
        super().__init__()
        self.supports_masking = True
        self.input_size = input_size  # api doc: H_in
        self.hidden_size = hidden_size  # api doc: H_out
        self.sequence_size = sequence_size  # api doc: L (sequence length)

        rnn_layer_class = get_from_registry(cell_type, rnn_layers_registry)

        # todo: need to discuss how to re-enable intializers, regularizer
        rnn_params = {
            #     'activation': activation,
            # 'nonlinearity': nonlinearity,
            #     'recurrent_activation': recurrent_activation,
            'num_layers': num_layers,
            'bias': use_bias,
            #     'kernel_initializer': weights_initializer,
            #     'recurrent_initializer': recurrent_initializer,
            #     'bias_initializer': bias_initializer,
            #     'unit_forget_bias': unit_forget_bias,
            #     'kernel_regularizer': weights_regularizer,
            #     'recurrent_regularizer': recurrent_regularizer,
            #     'bias_regularizer': bias_regularizer,
            #     'activity_regularizer': activity_regularizer,
            #     # 'kernel_constraint': weights_constraint,
            #     # 'recurrent_constraint': recurrent_constraint,
            #     # 'bias_constraint': bias_constraint,
            'dropout': dropout,
            'bidirectional': bidirectional,
            #     'recurrent_dropout': recurrent_dropout,
            #     'return_sequences': True,
            #     'return_state': True,
        }

        # current design is delegating to PyTorch RNN/GRU/LSTM layer
        # to do stacking and bidirectional based on the num_layers and
        # bidirectional parameter values.
        self.layers = rnn_layer_class(
            input_size, hidden_size,
            batch_first=True,
            **rnn_params
        )

    @property
    def input_shape(self) -> torch.Size:
        """ Returns the size of the input tensor without the batch dimension. """
        return torch.Size([self.sequence_size, self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        """ Returns the size of the output tensor without the batch dimension."""
        output_tensor = self.forward(torch.rand(2, *self.input_shape))
        # output tensor is 2-tuple(hidden, final_state)
        return output_tensor[0].size()[1:]

    def forward(self, inputs, training=None, mask=None):
        hidden, final_state = self.layers(inputs)

        # todo: confirm that this is correct handling
        # extract final state for the last layer in the torch stacked rnn
        if isinstance(final_state, tuple):
            # lstm cell type
            final_state = final_state[0][-1], final_state[1][-1]
        else:
            # rnn or gru cell type
            final_state = final_state[-1]

        # todo: determine if needed for torch implementation
        # for layer in self.layers:
        #     outputs = layer(hidden, training=training, mask=mask)
        #     hidden = outputs[0]
        #     final_state = outputs[1:]
        # if final_state and len(final_state) == 1:
        #     final_state = final_state[0]
        return hidden, final_state


#
# Ludwig Customizations to selected TFA classes
# to support use of sampled softmax loss function
#
class BasicDecoderOutput(
    collections.namedtuple('BasicDecoderOutput',
                           ('rnn_output', 'sample_id', 'projection_input'))):
    pass


class BasicDecoder: #(tfa.seq2seq.BasicDecoder):
    def _projection_input_size(self):
        return tf.TensorShape(self.cell.output_size)

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
            tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self.sampler.sample_ids_dtype,
            tf.nest.map_structure(lambda _: dtype,
                                  self._projection_input_size())
        )

    # Ludwig specific implementation of BasicDecoder.step() method
    def step(self, time, inputs, state, training=None, name=None):
        cell_outputs, cell_state = self.cell(inputs, state, training=training)
        cell_state = tf.nest.pack_sequence_as(state,
                                              tf.nest.flatten(cell_state))

        # get projection_inputs to compute sampled_softmax_cross_entropy_loss
        projection_inputs = cell_outputs

        if self.output_layer is not None:
            cell_outputs = self.output_layer(cell_outputs)
        sample_ids = self.sampler.sample(
            time=time, outputs=cell_outputs, state=cell_state)
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids,
                                     projection_inputs)

        return (outputs, next_state, next_inputs, finished)
