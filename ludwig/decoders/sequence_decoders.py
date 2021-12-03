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

import torch
import torch.functional as F
import torch.nn as nn

from ludwig.constants import SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder

logger = logging.getLogger(__name__)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
    def __init__(
        self,
        vocab_size,
        max_sequence_length=100,
        cell_type="rnn",
        state_size=256,
        embedding_size=64,
        beam_width=1,
        num_layers=1,
        attention=None,
        tied_embeddings=None,
        is_timeseries=False,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        reduce_input="sum",
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_rnn = DecoderRNN(state_size, vocab_size)

        self.GO_SYMBOL = self.vocab_size
        self.END_SYMBOL = 0

    def forward(self, inputs, targets=None, mask=None):
        batch_size = inputs.shape()[0]

        # Massage encoder_output to whatever shape is necessary to use decoders.
        # shape [batch_size, seq_size, state_size]
        encoder_outputs = inputs["hidden"]
        # form dependent on cell_type
        # lstm: list([batch_size, state_size], [batch_size, state_size])
        # rnn, gru: [batch_size, state_size]
        encoder_output_state = self.prepare_encoder_output_state(inputs)  # encoder_hidden

        # Tensor to store decoder outputs.
        logits = torch.zeros(batch_size, self.max_sequence_length, self.vocab_size)

        decoder_input = torch.tensor([[self.GO_SYMBOL]])
        decoder_hidden = encoder_output_state

        # Decode until max length. Break if a EOS token is encountered.
        sequence_length = targets.size()[1] if targets is not None else self.max_sequence_length
        for di in range(sequence_length):
            decoder_output, decoder_hidden, decoder_attention = self.attn_decoder_rnn(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Place predictions in a tensor holding predictions for each token.
            logits[di] = decoder_output

            # Determine inputs for next time step.
            if targets is not None:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                if decoder_input.item() == self.END_SYMBOL:
                    break
            else:
                # loss += criterion(decoder_output, target_tensor[di])
                decoder_input = targets[di]  # Teacher forcing


@register_decoder("tagger", [SEQUENCE, TEXT])
class SequenceTaggerDecoder(Decoder):
    pass
