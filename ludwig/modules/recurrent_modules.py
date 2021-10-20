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
from typing import Optional

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
            input_size: int = None,
            hidden_size: int = 256,
            cell_type: str = 'rnn',
            sequence_size: Optional[int] = None,
            num_layers: int = 1,
            bidirectional: bool = False,
            use_bias: bool = True,
            dropout: float = 0.0,
            **kwargs
    ):
        super().__init__()
        self.supports_masking = True
        self.input_size = input_size  # api doc: H_in
        self.hidden_size = hidden_size  # api doc: H_out
        self.sequence_size = sequence_size  # api doc: L (sequence length)

        rnn_layer_class = get_from_registry(cell_type, rnn_layers_registry)

        rnn_params = {
            'num_layers': num_layers,
            'bias': use_bias,
            'dropout': dropout,
            'bidirectional': bidirectional
        }

        # Delegate recurrent params to PyTorch's RNN/GRU/LSTM implementations.
        self.layers = rnn_layer_class(
            input_size, hidden_size,
            batch_first=True,
            **rnn_params
        )

    @property
    def input_shape(self) -> torch.Size:
        if self.sequence_size:
            return torch.Size([self.sequence_size, self.input_size])
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        if self.sequence_size:
            return torch.Size([self.sequence_size, self.hidden_size])
        return torch.Size([self.hidden_size])

    def forward(self, inputs, mask=None):
        hidden, final_state = self.layers(inputs)

        if isinstance(final_state, tuple):
            # lstm cell type
            final_state = final_state[0][-1], final_state[1][-1]
        else:
            # rnn or gru cell type
            final_state = final_state[-1]

        return hidden, final_state


#
# Ludwig Customizations to selected TFA classes
# to support use of sampled softmax loss function
#
class BasicDecoderOutput(
    collections.namedtuple('BasicDecoderOutput',
                           ('rnn_output', 'sample_id', 'projection_input'))):
    pass


class BasicDecoder:  # (tfa.seq2seq.BasicDecoder):
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
