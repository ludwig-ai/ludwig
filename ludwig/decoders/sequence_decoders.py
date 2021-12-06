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
from typing import Dict

import torch
import torch.nn as nn

from ludwig.constants import HIDDEN, LOGITS, SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder

logger = logging.getLogger(__name__)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        # Unsqueeze predicted token.
        input = input.unsqueeze(1).to(torch.int)

        output = self.relu(self.embedding(input))

        output, hidden = self.gru(output, hidden)

        # output = self.softmax(self.out(output))
        output = self.out(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
    def __init__(
        self,
        vocab_size,
        max_sequence_length=100,
        cell_type="gru",
        input_size=256,
        num_layers=1,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_rnn = DecoderRNN(input_size, vocab_size)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: Dict[str, torch.Tensor], target=None, mask=None):
        # inputs[HIDDEN]: [batch_size, seq_size, state_size]
        encoder_output_state = inputs[HIDDEN]
        batch_size = encoder_output_state.size()[0]

        # Tensor to store decoder outputs.
        logits = torch.zeros(batch_size, self.max_sequence_length, self.vocab_size)

        # TODO: Use real go symbol.
        # decoder_input = target[:, 0]
        decoder_input = torch.ones([batch_size])

        # Unsqueeze to account for extra dimension for multilayer layer dimension.
        decoder_hidden = encoder_output_state.unsqueeze(0)

        # Decode until max length.
        for di in range(self.max_sequence_length):
            decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)

            # Holding logits for each token.
            # decoder_output: [batch_size, 1, vocab_size]
            logits[:, di, :] = decoder_output.data.squeeze(1)

            # Determine inputs for next time step.
            # TODO: Flip a coin for not using teacher forcing during training.
            if target is None:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
            else:
                # Teacher forcing
                decoder_input = target[:, di]

        # TODO: Is projection input important?
        return {LOGITS: logits}


@register_decoder("tagger", [SEQUENCE, TEXT])
class SequenceTaggerDecoder(Decoder):
    pass
