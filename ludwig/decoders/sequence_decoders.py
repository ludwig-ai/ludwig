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

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOGITS, PREDICTIONS, PROBABILITIES, SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.decoders.sequence_decoder_utils import get_lstm_init_state, get_rnn_init_state
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.decoders.sequence_decoders import SequenceGeneratorDecoderConfig
from ludwig.utils import strings_utils

logger = logging.getLogger(__name__)


@DeveloperAPI
class RNNDecoder(nn.Module):
    """GRU or RNN-based decoder."""

    def __init__(self, hidden_size: int, vocab_size: int, cell_type: str, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if cell_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
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
            - output: [batch_size, 1, vocab_size] tensor with the logits.
            - hidden: [num_layers, batch_size, hidden_size] tensor with the hidden state for the next time step.
        """
        # Unsqueeze predicted tokens.
        input = input.unsqueeze(1).to(torch.int)
        output = self.embedding(input)
        output, hidden = self.rnn(output, hidden)
        output_logits = self.out(output)
        return output_logits, hidden


@DeveloperAPI
class LSTMDecoder(nn.Module):
    """LSTM-based decoder."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
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


@DeveloperAPI
class SequenceRNNDecoder(nn.Module):
    """RNN-based decoder over multiple time steps."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_sequence_length: int,
        cell_type: str,
        num_layers: int = 1,
        reduce_input="sum",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rnn_decoder = RNNDecoder(hidden_size, vocab_size, cell_type, num_layers=num_layers)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)
        self.num_layers = num_layers

        self.register_buffer("logits", torch.zeros([max_sequence_length, vocab_size]))
        self.register_buffer("decoder_input", torch.Tensor([strings_utils.SpecialSymbol.START.value]))

    def forward(self, combiner_outputs: Dict[str, torch.Tensor], target: torch.Tensor):
        """Runs max_sequence_length RNN decoding time steps.

        Args:
            combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with target symbols.

        Returns:
            Tensor of logits [batch_size, max_sequence_length, vocab_size].
        """
        # Prepare the encoder output state.
        decoder_hidden = get_rnn_init_state(combiner_outputs, self.reduce_sequence, self.num_layers)

        batch_size = decoder_hidden.size()[1]

        # Tensor to store decoder output logits.
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1, 1)

        # Initialize the decoder with start symbols.
        decoder_input = self.decoder_input.repeat(batch_size)

        # Unsqueeze to account for extra multilayer dimension.
        # decoder_hidden = encoder_output_state.unsqueeze(0)

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
                # Squeeze out multilayer and vocabulary dimensions.
                decoder_input = topi.squeeze(1).squeeze(1).detach()  # detach from history as input
            else:
                # Teacher forcing.
                decoder_input = target[:, di]

        return logits


@DeveloperAPI
class SequenceLSTMDecoder(nn.Module):
    """LSTM-based decoder over multiple time steps."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_sequence_length: int,
        reduce_input: str = "sum",
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm_decoder = LSTMDecoder(hidden_size, vocab_size, num_layers)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)
        self.num_layers = num_layers

        self.register_buffer("logits", torch.zeros([max_sequence_length, vocab_size]))
        self.register_buffer("decoder_input", torch.Tensor([strings_utils.SpecialSymbol.START.value]))

    def forward(self, combiner_outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Runs max_sequence_length LSTM decoding time steps.

        Args:
            combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with target symbols.

        Returns:
            Tensor of logits [batch_size, max_sequence_length, vocab_size].
        """
        # Prepare the decoder initial state.
        decoder_hidden, decoder_cell_state = get_lstm_init_state(
            combiner_outputs, self.reduce_sequence, self.num_layers
        )
        batch_size = decoder_hidden.size()[1]

        # Initialize the decoder with start symbols.
        decoder_input = self.decoder_input.repeat(batch_size)

        # Tensor to store decoder output logits.
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1, 1)

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
                # Squeeze out multilayer and vocabulary dimensions.
                decoder_input = topi.squeeze(1).squeeze(1).detach()  # detach from history as input
            else:
                # Teacher forcing.
                decoder_input = target[:, di]

        return logits


@DeveloperAPI
@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
    """Dispatcher for different sequence generator decoders."""

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        cell_type: str = "gru",
        input_size: int = 256,
        reduce_input: str = "sum",
        num_layers: int = 1,
        decoder_config=None,
        **kwargs,
    ):
        """
        Args:
            vocab_size: Vocab size.
            max_sequence_length: Maximum sequence length.
            cell_type: Type of RNN cell to use. 'rnn', 'gru', or 'lstm'.
            input_size: Size of incoming combiner output.
            reduce_input: Mode with which to reduce incoming combiner output, if needed.
            num_layers: Number of layers for the RNN deecoders.
        """
        super().__init__()
        self.config = decoder_config

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        if cell_type == "lstm":
            self.rnn_decoder = SequenceLSTMDecoder(
                hidden_size=input_size,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                reduce_input=reduce_input,
                num_layers=num_layers,
            )
        else:
            self.rnn_decoder = SequenceRNNDecoder(
                hidden_size=input_size,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                cell_type=cell_type,
                reduce_input=reduce_input,
                num_layers=num_layers,
            )

    def forward(
        self, combiner_outputs: Dict[str, torch.Tensor], target: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Decodes combiner_outputs into a sequence.

        Args:
            combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with target symbols.

        Returns:
            Dictionary of tensors of logits [batch_size, max_sequence_length, vocab_size].
        """
        logits = self.rnn_decoder(combiner_outputs, target)
        return {LOGITS: logits}

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    @staticmethod
    def get_schema_cls():
        return SequenceGeneratorDecoderConfig

    @property
    def input_shape(self):
        # Dummy implementation.
        return torch.Size([1])

    @property
    def output_shape(self):
        return torch.Size([self.max_sequence_length, self.vocab_size])
