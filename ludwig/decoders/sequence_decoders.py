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
from ludwig.modules.reduction_modules import SequenceReducer
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
    """RNN-based decoder over multiple time steps."""

    def __init__(self, hidden_size: int, vocab_size: int, max_sequence_length: int, cell_type: str, reduce_input="sum"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rnn_decoder = RNNDecoder(hidden_size, vocab_size, cell_type)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)

        self.register_buffer("logits", torch.zeros([max_sequence_length, vocab_size]))
        self.register_buffer("decoder_input", torch.Tensor([strings_utils.SpecialSymbol.START.value]))

    def forward(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor):
        """Runs max_sequence_length RNN decoding time steps.

        Args:
            input: [batch_size] tensor with the previous step's predicted symbol.
            hidden_state: [batch_size, hidden_size] tensor with the previous step's hidden state.
            cell_state: [batch_size, hidden_size] tensor with the previous step's cell state.

        Returns:
            Tensor of logits [batch_size, max_sequence_length, vocab_size].
        """
        # Prepare the encoder output state.
        encoder_output_state = self.prepare_encoder_output_state(inputs)

        batch_size = encoder_output_state.size()[0]

        # Tensor to store decoder output logits.
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1, 1)

        # Initialize the decoder with start symbols.
        decoder_input = self.decoder_input.repeat(batch_size)

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

    def prepare_encoder_output_state(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the hidden state that the RNN decoder should start with.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.

        Returns:
            Tensor of [batch_size, hidden_size].
        """
        if "encoder_output_state" not in inputs:
            hidden = inputs[HIDDEN]
            if len(hidden.size()) == 3:
                # Reduce to [batch_size, hidden_size]
                return self.reduce_sequence(hidden)
            if len(hidden.size()) == 2:
                return hidden
            raise ValueError("Only works for 1d or 2d encoder_output")

        encoder_output_state = inputs["encoder_output_state"]
        if not isinstance(encoder_output_state, tuple):
            # RNN encoder.
            return encoder_output_state
        if len(encoder_output_state) == 2:
            # LSTM encoder. Use the hidden state and ignore the cell state.
            return encoder_output_state[0]
        if len(encoder_output_state) == 4:
            # Bi-directional LSTM encoder. Use the average of hidden states and ignore cell state.
            return torch.mean([encoder_output_state[0], encoder_output_state[2]])

        raise ValueError(
            f"Invalid sequence decoder inputs with keys: {inputs.keys()} with extracted encoder state: "
            + f"{encoder_output_state} that was invalid. Please double check the compatibility of your encoder and "
            + "decoder."
        )


class SequenceLSTMDecoder(nn.Module):
    """LSTM-based decoder over multiple time steps."""

    def __init__(self, hidden_size: int, vocab_size: int, max_sequence_length: int, reduce_input="sum"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm_decoder = LSTMDecoder(hidden_size, vocab_size)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)

        self.register_buffer("logits", torch.zeros([max_sequence_length, vocab_size]))
        self.register_buffer("decoder_input", torch.Tensor([strings_utils.SpecialSymbol.START.value]))

    def forward(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Runs max_sequence_length LSTM decoding time steps.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with target symbols.

        Returns:
            Tensor of logits [batch_size, max_sequence_length, vocab_size].
        """
        # Prepare the encoder output state.
        decoder_hidden, decoder_cell_state = self.prepare_encoder_output_state(inputs)

        batch_size = decoder_hidden.size()[0]

        # Unsqueeze to account for extra multilayer dimension.
        # TODO(Justin): Support multi-layer LSTM decoders.
        decoder_hidden = decoder_hidden.unsqueeze(0)
        decoder_cell_state = decoder_cell_state.unsqueeze(0)

        # Tensor to store decoder output logits.
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1, 1)

        # Initialize the decoder with start symbols.
        decoder_input = self.decoder_input.repeat(batch_size)

        # Decode until max length.
        for di in range(self.max_sequence_length):
            print(f"Incoming decoder_input: {decoder_input.size()}")
            print(f"Incoming decoder_hidden: {decoder_hidden.size()}")
            print(f"Incoming decoder_cell_state: {decoder_cell_state.size()}")
            decoder_output, decoder_hidden, decoder_cell_state = self.lstm_decoder(
                decoder_input, decoder_hidden, decoder_cell_state
            )
            print(f"Outgoing decoder_input: {decoder_input.size()}")
            print(f"Outgoing decoder_hidden: {decoder_hidden.size()}")
            print(f"Outgoing decoder_cell_state: {decoder_cell_state.size()}")

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

    def prepare_encoder_output_state(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the states that the LSTM decoder should start with.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.

        Returns:
            Tuple of 2 tensors (decoder hidden state, decoder cell state), each [batch_size, hidden_size].
        """
        if "encoder_output_state" not in inputs:
            hidden = inputs[HIDDEN]
            if len(hidden.size()) > 3:
                raise ValueError(
                    f"Encoder hidden state passed to LSTM decoder has {len(hidden.size())} dimensions, which is too "
                    + "many. Please check that the encoder returns 1d or 2d output."
                )
            encoder_output_state = hidden
            if len(hidden.size()) == 3:
                encoder_output_state = self.reduce_sequence(hidden)
            return (encoder_output_state, encoder_output_state)

        encoder_output_state = inputs["encoder_output_state"]
        if not isinstance(encoder_output_state, tuple):
            return (encoder_output_state, encoder_output_state)

        if len(encoder_output_state) == 2:
            # The encoder was probably an LSTM.
            return encoder_output_state

        if len(encoder_output_state) == 4:
            # The encoder was probably a bi-LSTM.
            # Use the average of the encoder's hidden states for hidden state.
            # Use the average of the encoder's cell states for cell state.
            return (
                torch.mean([encoder_output_state[0], encoder_output_state[2]]),
                torch.mean([encoder_output_state[1], encoder_output_state[3]]),
            )

        raise ValueError(
            f"Invalid sequence decoder inputs with keys: {inputs.keys()} with extracted encoder state: "
            + f"{encoder_output_state} that was invalid. Please double check the compatibility of your encoder and "
            + "decoder."
        )


@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
    """Dispatcher for different sequence generator decoders."""

    def __init__(
        self,
        vocab_size,
        max_sequence_length=100,
        cell_type="gru",
        input_size=256,
        reduce_input="sum",
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        if cell_type == "lstm":
            self.rnn_decoder = SequenceLSTMDecoder(input_size, vocab_size, max_sequence_length, reduce_input)
        else:
            self.rnn_decoder = SequenceRNNDecoder(input_size, vocab_size, max_sequence_length, cell_type, reduce_input)

    def forward(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Decodes the inputs into a sequence.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with target symbols.

        Returns:
            Dictionary of tensors of logits [batch_size, max_sequence_length, vocab_size].
        """
        logits = self.rnn_decoder(inputs, target)
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
