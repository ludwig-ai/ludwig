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
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ludwig.constants import HIDDEN, LOGITS, SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.utils import strings_utils

logger = logging.getLogger(__name__)


class RNNDecoder(nn.Module):
    """GRU or RNN-based decoder."""

    def __init__(self, hidden_size: int, vocab_size: int, cell_type: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if cell_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

        # Have the embedding and projection share weights.
        # This is a trick used by the Transformer, and seems to attain better loss.
        # See section 3.4 of https://arxiv.org/pdf/1706.03762.pdf.
        self.out.weight = self.embedding.weight

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a single decoding time step.

        Modeled off of https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.

        Args:
            input: [batch_size] tensor with the previous step's predicted symbol.
            hidden: [batch_size, hidden_size] tensor with the previous step's hidden state.

        Returns:
            Tuple of two tensors:
            - output: [batch_size, vocab_size] tensor with the logits.
            - hidden: [batch_size, hidden_size] tensor with the hidden state for the next time step.
        """
        # Unsqueeze predicted tokens.
        input = input.unsqueeze(1).to(torch.int)
        output = self.embedding(input)
        output, hidden = self.rnn(output, hidden)
        output_logits = self.out(output)
        return output_logits, hidden


class LSTMDecoder(nn.Module):
    """LSTM-based decoder."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

        # Have the embedding and projection share weights.
        # This is a trick used by the Transformer, and seems to attain better loss.
        # See section 3.4 of https://arxiv.org/pdf/1706.03762.pdf.
        self.out.weight = self.embedding.weight

    def forward(
        self, input: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs a single decoding time step.

        Modeled off of https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.

        Args:
            input: [batch_size] tensor with the previous step's predicted symbol.
            hidden_state: [batch_size, hidden_size] tensor with the previous step's hidden state.
            cell_state: [batch_size, hidden_size] tensor with the previous step's cell state.

        Returns:
            Tuple of 3 tensors:
            - output: [batch_size, vocab_size] tensor with the logits.
            - hidden_state: [batch_size, hidden_size] tensor with the hidden state for the next time step.
            - cell_state: [batch_size, hidden_size] tensor with the cell state for the next time step.
        """
        # Unsqueeze predicted tokens.
        input = input.unsqueeze(1).to(torch.int)
        output = self.embedding(input)
        output, (hidden_state, cell_state) = self.lstm(output, (hidden_state, cell_state))
        output_logits = self.out(output)
        return output_logits, hidden_state, cell_state


class SequenceRNNDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, max_sequence_length: int, cell_type: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rnn_decoder = RNNDecoder(hidden_size, vocab_size, cell_type)
        self.max_sequence_length = max_sequence_length

    def forward(self, encoder_output_state: torch.Tensor, target: torch.Tensor):
        batch_size = encoder_output_state.size()[0]

        # Tensor to store decoder output logits.
        logits = torch.zeros(batch_size, self.max_sequence_length, self.vocab_size)

        # Initialize the decoder with start symbols.
        decoder_input = torch.empty(batch_size).fill_(strings_utils.START_IDX)

        # Unsqueeze to account for extra multilayer dimension.
        decoder_hidden = encoder_output_state.unsqueeze(0)

        # Decode until max length.
        for di in range(self.max_sequence_length):
            decoder_output, decoder_hidden = self.rnn_decoder(decoder_input, decoder_hidden)

            # decoder_output: [batch_size, 1, vocab_size]
            # Squeeze out the multilayer dimension and save logits.
            logits[:, di, :] = decoder_output.squeeze(1)

            # Determine inputs for next time step.
            # Using teacher forcing causes the model to converge faster but when the trained network is exploited, it
            # may be unstable: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf.
            # TODO: Use a configurable ratio for how often to use teacher forcing during training.
            if target is None:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
            else:
                # Teacher forcing.
                decoder_input = target[:, di]

        return logits


class SequenceLSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, max_sequence_length: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm_decoder = LSTMDecoder(hidden_size, vocab_size)
        self.max_sequence_length = max_sequence_length

    def forward(self, encoder_output_state: torch.Tensor, target: torch.Tensor):
        batch_size = encoder_output_state.size()[0]

        # Tensor to store decoder output logits.
        logits = torch.zeros(batch_size, self.max_sequence_length, self.vocab_size)

        # Initialize the decoder with start symbols.
        decoder_input = torch.empty(batch_size).fill_(strings_utils.START_IDX)

        # Unsqueeze to account for extra multilayer dimension.
        decoder_hidden = encoder_output_state.unsqueeze(0)

        # Initialize the cell state to the encoder's output state.
        decoder_cell_state = decoder_hidden

        # Decode until max length.
        for di in range(self.max_sequence_length):
            decoder_output, decoder_hidden, decoder_cell_state = self.lstm_decoder(
                decoder_input, decoder_hidden, decoder_cell_state
            )

            # decoder_output: [batch_size, 1, vocab_size]
            # Squeeze out the multilayer dimension and save logits.
            logits[:, di, :] = decoder_output.squeeze(1)

            # Determine inputs for next time step.
            # Using teacher forcing causes the model to converge faster but when the trained network is exploited, it
            # may be unstable: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf.
            # TODO: Use a configurable ratio for how often to use teacher forcing during training.
            if target is None:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
            else:
                # Teacher forcing.
                decoder_input = target[:, di]

        return logits


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
        if cell_type == "lstm":
            self.rnn_decoder = SequenceLSTMDecoder(input_size, vocab_size, max_sequence_length)
        else:
            self.rnn_decoder = SequenceRNNDecoder(input_size, vocab_size, max_sequence_length, cell_type)
        self.max_sequence_length = max_sequence_length

    def forward(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Decodes the inputs into a sequence.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with target symbols.

        Returns:
            Dictionary of tensors of decoder outputs like logits [batch_size, max_sequence_length, vocab_size].
        """
        encoder_output_state = inputs[HIDDEN]
        logits = self.rnn_decoder(encoder_output_state, target)
        return {LOGITS: logits}

    def get_prediction_set(self):
        return {LOGITS}

    @property
    def input_shape(self):
        # Dummy implementation.
        return torch.Size([1])

    @property
    def output_shape(self):
        return torch.Size([self.max_sequence_length, self.vocab_size])


@register_decoder("tagger", [SEQUENCE, TEXT])
class SequenceTaggerDecoder(Decoder):
    pass
