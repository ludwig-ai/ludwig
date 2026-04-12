# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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
import math

import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import LOGITS, PREDICTIONS, PROBABILITIES, SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.decoders.sequence_decoder_utils import get_lstm_init_state, get_rnn_init_state
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.decoders.sequence_decoders import SequenceGeneratorDecoderConfig, TransformerDecoderConfig
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

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs a single decoding time step.

        Args:
            input: [batch_size] tensor with the previous step's predicted symbol.
            hidden_state: [num_layers, batch_size, hidden_size] tensor.
            cell_state: [num_layers, batch_size, hidden_size] tensor.

        Returns:
            Tuple of 3 tensors:
            - output: [batch_size, vocab_size] tensor with the logits.
            - hidden_state: [num_layers, batch_size, hidden_size] tensor.
            - cell_state: [num_layers, batch_size, hidden_size] tensor.
        """
        input = input.unsqueeze(1).to(torch.int)
        output = self.embedding(input)
        output, (hidden_state, cell_state) = self.lstm(output, (hidden_state, cell_state))
        output_logits = self.out(output)
        return output_logits, hidden_state, cell_state


def _teacher_forcing_prob(step: int, decay: str, decay_rate: float) -> float:
    """Computes the teacher-forcing probability for a given decoding step.

    Implements scheduled sampling schedules from Bengio et al., NeurIPS 2015:
    "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks".

    Args:
        step: Current decoding step index (0-based).
        decay: One of 'none', 'linear', or 'exponential'.
        decay_rate: Per-step decay rate.

    Returns:
        Probability in [0, 1] of using the ground-truth token as next input.
    """
    if decay == "none":
        return 1.0
    if decay == "linear":
        return max(0.0, 1.0 - decay_rate * step)
    if decay == "exponential":
        return math.exp(-decay_rate * step)
    raise ValueError(f"Unknown teacher_forcing_decay: {decay!r}. Choose 'none', 'linear', or 'exponential'.")


@DeveloperAPI
class SequenceRNNDecoder(nn.Module):
    """RNN-based decoder over multiple time steps.

    Supports scheduled sampling (Bengio et al., NeurIPS 2015) to smoothly
    interpolate between teacher forcing during early training and fully
    autoregressive decoding at inference.

    References:
        Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015).
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
        NeurIPS 2015. https://arxiv.org/abs/1506.03099

    When to use:
        - Sequence/text generation with a simple RNN backbone.
        - Use teacher_forcing_decay='linear' or 'exponential' when the model
          overfits to teacher forcing and performs poorly at inference.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_sequence_length: int,
        cell_type: str,
        num_layers: int = 1,
        reduce_input="sum",
        teacher_forcing_decay: str = "none",
        teacher_forcing_decay_rate: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rnn_decoder = RNNDecoder(hidden_size, vocab_size, cell_type, num_layers=num_layers)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)
        self.num_layers = num_layers
        self.teacher_forcing_decay = teacher_forcing_decay
        self.teacher_forcing_decay_rate = teacher_forcing_decay_rate

        self.register_buffer("logits", torch.zeros([max_sequence_length, vocab_size]))
        self.register_buffer("decoder_input", torch.Tensor([strings_utils.SpecialSymbol.START.value]))

    def forward(self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor):
        """Runs max_sequence_length RNN decoding time steps.

        Args:
            combiner_outputs: Dict of tensors from the combiner and other output features.
            target: [batch_size, max_sequence_length] target symbols, or None at inference.

        Returns:
            Tensor of logits [batch_size, max_sequence_length, vocab_size].
        """
        decoder_hidden = get_rnn_init_state(combiner_outputs, self.reduce_sequence, self.num_layers)
        batch_size = decoder_hidden.size()[1]
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1, 1)
        decoder_input = self.decoder_input.repeat(batch_size)
        is_training = self.training and target is not None

        for di in range(self.max_sequence_length):
            decoder_output, decoder_hidden = self.rnn_decoder(decoder_input, decoder_hidden)
            logits[:, di, :] = decoder_output.squeeze(1)

            if target is None:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).squeeze(1).detach()
            elif is_training and self.teacher_forcing_decay != "none":
                tf_prob = _teacher_forcing_prob(di, self.teacher_forcing_decay, self.teacher_forcing_decay_rate)
                if torch.rand(1).item() < tf_prob:
                    decoder_input = target[:, di]
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(1).squeeze(1).detach()
            else:
                decoder_input = target[:, di]

        return logits


@DeveloperAPI
class SequenceLSTMDecoder(nn.Module):
    """LSTM-based decoder over multiple time steps.

    Supports scheduled sampling (Bengio et al., NeurIPS 2015) to smoothly
    interpolate between teacher forcing during early training and fully
    autoregressive decoding at inference.

    References:
        Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015).
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
        NeurIPS 2015. https://arxiv.org/abs/1506.03099

    When to use:
        - Sequence/text generation tasks where long-range dependencies matter.
        - Use teacher_forcing_decay='linear' or 'exponential' when the model
          overfits to teacher forcing and performs poorly at inference.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_sequence_length: int,
        reduce_input: str = "sum",
        num_layers: int = 1,
        teacher_forcing_decay: str = "none",
        teacher_forcing_decay_rate: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm_decoder = LSTMDecoder(hidden_size, vocab_size, num_layers)
        self.max_sequence_length = max_sequence_length
        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)
        self.num_layers = num_layers
        self.teacher_forcing_decay = teacher_forcing_decay
        self.teacher_forcing_decay_rate = teacher_forcing_decay_rate

        self.register_buffer("logits", torch.zeros([max_sequence_length, vocab_size]))
        self.register_buffer("decoder_input", torch.Tensor([strings_utils.SpecialSymbol.START.value]))

    def forward(self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Runs max_sequence_length LSTM decoding time steps.

        Args:
            combiner_outputs: Dict of tensors from the combiner and other output features.
            target: [batch_size, max_sequence_length] target symbols, or None at inference.

        Returns:
            Tensor of logits [batch_size, max_sequence_length, vocab_size].
        """
        decoder_hidden, decoder_cell_state = get_lstm_init_state(
            combiner_outputs, self.reduce_sequence, self.num_layers
        )
        batch_size = decoder_hidden.size()[1]
        decoder_input = self.decoder_input.repeat(batch_size)
        logits = self.logits.unsqueeze(0).repeat(batch_size, 1, 1)
        is_training = self.training and target is not None

        for di in range(self.max_sequence_length):
            decoder_output, decoder_hidden, decoder_cell_state = self.lstm_decoder(
                decoder_input, decoder_hidden, decoder_cell_state
            )
            logits[:, di, :] = decoder_output.squeeze(1)

            if target is None:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).squeeze(1).detach()
            elif is_training and self.teacher_forcing_decay != "none":
                tf_prob = _teacher_forcing_prob(di, self.teacher_forcing_decay, self.teacher_forcing_decay_rate)
                if torch.rand(1).item() < tf_prob:
                    decoder_input = target[:, di]
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(1).squeeze(1).detach()
            else:
                decoder_input = target[:, di]

        return logits


def _beam_search_stateful(
    step_fn,
    init_state,
    state_index_fn,
    start_token: int,
    max_sequence_length: int,
    vocab_size: int,
    beam_width: int,
    length_penalty: float,
    device: torch.device,
):
    """Beam search with an arbitrary stateful step function.

    Args:
        step_fn: (tokens [beam_width], state) -> (log_probs [beam_width, vocab], new_state).
        init_state: Initial state sized for beam_width beams.
        state_index_fn: (state, beam_indices) -> reordered state.
        start_token: Vocabulary index of the start token.
        max_sequence_length: Maximum number of tokens to generate.
        vocab_size: Vocabulary size.
        beam_width: Number of beams.
        length_penalty: Exponent for length normalisation.
        device: Torch device.

    Returns:
        sequences: [beam_width, max_sequence_length] best sequences (highest score first).
        scores: [beam_width] length-normalised log-probability scores.
    """
    sequences = torch.full((beam_width, 1), start_token, dtype=torch.long, device=device)
    cum_log_probs = torch.zeros(beam_width, device=device)
    state = init_state

    for _step in range(max_sequence_length):
        last_tokens = sequences[:, -1]
        log_probs, state = step_fn(last_tokens, state)

        total_log_probs = cum_log_probs.unsqueeze(1) + log_probs
        flat = total_log_probs.view(-1)

        top_log_probs, top_idx = flat.topk(beam_width)
        beam_idx = top_idx // vocab_size
        token_idx = top_idx % vocab_size

        sequences = torch.cat([sequences[beam_idx], token_idx.unsqueeze(1)], dim=1)
        cum_log_probs = top_log_probs
        state = state_index_fn(state, beam_idx)

    lengths = torch.full((beam_width,), max_sequence_length, dtype=torch.float, device=device)
    scores = cum_log_probs / lengths.pow(length_penalty)
    order = scores.argsort(descending=True)
    return sequences[order, 1:], scores[order]


@DeveloperAPI
@register_decoder("generator", [SEQUENCE, TEXT])
class SequenceGeneratorDecoder(Decoder):
    """Dispatcher for different sequence generator decoders.

    Supports three RNN cell types (gru, lstm, rnn) plus:

    - Scheduled sampling (Bengio et al., NeurIPS 2015): set teacher_forcing_decay
      to 'linear' or 'exponential' to gradually shift from teacher forcing to
      model predictions during training.

    - Beam search: set beam_width > 1 to enable beam search at inference time.
      The beam_length_penalty controls length normalisation of beam scores.

    References:
        Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015).
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
        NeurIPS 2015. https://arxiv.org/abs/1506.03099

    When to use:
        - Classic sequence generation with RNN backbones.
        - Use cell_type='gru' (default) for most tasks.
        - Use cell_type='lstm' when long-range dependencies are important.
        - Use beam_width > 1 for higher-quality outputs at inference time.
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        cell_type: str = "gru",
        input_size: int = 256,
        reduce_input: str = "sum",
        num_layers: int = 1,
        teacher_forcing_decay: str = "none",
        teacher_forcing_decay_rate: float = 0.01,
        beam_width: int = 1,
        beam_length_penalty: float = 1.0,
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
            num_layers: Number of layers for the RNN decoders.
            teacher_forcing_decay: Schedule for teacher forcing probability.
            teacher_forcing_decay_rate: Per-step decay rate for teacher forcing probability.
            beam_width: Beam width for inference (1 = greedy).
            beam_length_penalty: Length penalty exponent for beam search scoring.
        """
        super().__init__()
        self.config = decoder_config

        self.vocab_size = vocab_size
        self.input_size = input_size
        self.max_sequence_length = max_sequence_length
        self.beam_width = beam_width
        self.beam_length_penalty = beam_length_penalty

        if cell_type == "lstm":
            self.rnn_decoder = SequenceLSTMDecoder(
                hidden_size=input_size,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                reduce_input=reduce_input,
                num_layers=num_layers,
                teacher_forcing_decay=teacher_forcing_decay,
                teacher_forcing_decay_rate=teacher_forcing_decay_rate,
            )
        else:
            self.rnn_decoder = SequenceRNNDecoder(
                hidden_size=input_size,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                cell_type=cell_type,
                reduce_input=reduce_input,
                num_layers=num_layers,
                teacher_forcing_decay=teacher_forcing_decay,
                teacher_forcing_decay_rate=teacher_forcing_decay_rate,
            )

    def _greedy_decode(self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        return self.rnn_decoder(combiner_outputs, target)

    def _beam_decode(self, combiner_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Beam search decoding. Returns logits [batch_size, max_sequence_length, vocab_size].

        Since beam search selects hard token indices, the returned tensor has 1.0 at the chosen token and 0.0 elsewhere
        so downstream argmax produces correct predictions.
        """
        inner = self.rnn_decoder
        is_lstm = isinstance(inner, SequenceLSTMDecoder)
        device = next(self.parameters()).device

        if is_lstm:
            h, c = get_lstm_init_state(combiner_outputs, inner.reduce_sequence, inner.num_layers)
            batch_size = h.size(1)
        else:
            h = get_rnn_init_state(combiner_outputs, inner.reduce_sequence, inner.num_layers)
            batch_size = h.size(1)

        start_token = int(strings_utils.SpecialSymbol.START.value)
        all_logits = []

        for b in range(batch_size):
            if is_lstm:
                h_b = h[:, b : b + 1, :].expand(-1, self.beam_width, -1).contiguous()
                c_b = c[:, b : b + 1, :].expand(-1, self.beam_width, -1).contiguous()

                def step_fn_lstm(tokens, state):
                    h_s, c_s = state
                    logits_out, h_new, c_new = inner.lstm_decoder(tokens, h_s, c_s)
                    return torch.log_softmax(logits_out.squeeze(1), dim=-1), (h_new, c_new)

                def state_index_lstm(state, idx):
                    h_s, c_s = state
                    return h_s[:, idx, :].unsqueeze(1), c_s[:, idx, :].unsqueeze(1)

                seqs, _ = _beam_search_stateful(
                    step_fn_lstm,
                    (h_b, c_b),
                    state_index_lstm,
                    start_token,
                    self.max_sequence_length,
                    self.vocab_size,
                    self.beam_width,
                    self.beam_length_penalty,
                    device,
                )
            else:
                h_b = h[:, b : b + 1, :].expand(-1, self.beam_width, -1).contiguous()

                def step_fn_rnn(tokens, state):
                    logits_out, h_new = inner.rnn_decoder(tokens, state)
                    return torch.log_softmax(logits_out.squeeze(1), dim=-1), h_new

                def state_index_rnn(state, idx):
                    return state[:, idx, :].unsqueeze(1)

                seqs, _ = _beam_search_stateful(
                    step_fn_rnn,
                    h_b,
                    state_index_rnn,
                    start_token,
                    self.max_sequence_length,
                    self.vocab_size,
                    self.beam_width,
                    self.beam_length_penalty,
                    device,
                )

            best_seq = seqs[0]
            item_logits = torch.zeros(self.max_sequence_length, self.vocab_size, device=device)
            for t in range(self.max_sequence_length):
                item_logits[t, best_seq[t]] = 1.0
            all_logits.append(item_logits)

        return torch.stack(all_logits, dim=0)

    def forward(
        self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """Decodes combiner_outputs into a sequence.

        During training (target provided): uses teacher forcing with optional scheduled sampling.
        During inference (target=None): greedy decoding when beam_width=1, beam search otherwise.

        Args:
            combiner_outputs: Dict of tensors from the combiner and other output features.
            target: [batch_size, max_sequence_length] target symbols.

        Returns:
            Dict with LOGITS -> [batch_size, max_sequence_length, vocab_size].
        """
        if target is None and self.beam_width > 1:
            logits = self._beam_decode(combiner_outputs)
        else:
            logits = self._greedy_decode(combiner_outputs, target)
        return {LOGITS: logits}

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    @staticmethod
    def get_schema_cls():
        return SequenceGeneratorDecoderConfig

    @property
    def input_shape(self):
        return torch.Size([1])

    @property
    def output_shape(self):
        return torch.Size([self.max_sequence_length, self.vocab_size])


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., NeurIPS 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)].unsqueeze(1)
        return self.dropout(x)


@DeveloperAPI
@register_decoder("transformer_generator", [SEQUENCE, TEXT])
class SequenceTransformerDecoder(Decoder):
    """Transformer-based autoregressive sequence/text decoder.

    Uses nn.TransformerDecoder with cross-attention over the encoder memory.
    Training uses parallel teacher forcing for efficiency; inference is
    autoregressive (greedy or beam search).

    Architecture:
        encoder output -> [optional linear projection] -> cross-attention memory
        target tokens  -> embedding + positional encoding
                       -> N x TransformerDecoderLayer (self-attn, cross-attn, FFN)
                       -> Linear (d_model -> vocab_size) [weight-tied to embedding]

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
        Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
        Attention Is All You Need. NeurIPS 2017.
        https://arxiv.org/abs/1706.03762

    When to use:
        - Prefer over RNN decoders when the encoder produces rich contextual
          representations (e.g. BERT-style) or for sequences longer than ~30 tokens.
        - Set beam_width > 1 to improve output quality at inference time.
        - Increase num_layers / num_heads for more expressive capacity.
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        input_size: int = 256,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_size: int = 1024,
        dropout: float = 0.1,
        reduce_input: str = "sum",
        beam_width: int = 1,
        beam_length_penalty: float = 1.0,
        decoder_config=None,
        **kwargs,
    ):
        """
        Args:
            vocab_size: Target vocabulary size.
            max_sequence_length: Maximum number of tokens to generate.
            input_size: Dimension of the encoder output (memory).
            d_model: Internal transformer dimension. A projection is inserted if input_size != d_model.
            num_layers: Number of TransformerDecoderLayer stacks.
            num_heads: Number of attention heads. d_model % num_heads must equal 0.
            ffn_size: Feed-forward network hidden size.
            dropout: Dropout probability.
            reduce_input: How to reduce a 3-D encoder output to 2-D when needed.
            beam_width: Beam width for inference (1 = greedy).
            beam_length_penalty: Length penalty exponent for beam search scoring.
        """
        super().__init__()
        self.config = decoder_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.beam_width = beam_width
        self.beam_length_penalty = beam_length_penalty

        self.memory_projection = nn.Linear(input_size, d_model) if input_size != d_model else None
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, max_len=max_sequence_length + 1, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.embedding.weight  # weight tying

        self.reduce_sequence = SequenceReducer(reduce_mode=reduce_input)
        self.register_buffer("start_token", torch.tensor([strings_utils.SpecialSymbol.START.value], dtype=torch.long))

    def _get_memory(self, combiner_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns encoder memory [batch_size, src_len, d_model] for cross-attention."""
        from ludwig.constants import HIDDEN

        hidden = combiner_outputs.get("encoder_output", combiner_outputs.get(HIDDEN))
        if hidden is None:
            raise ValueError("SequenceTransformerDecoder requires 'encoder_output' or 'hidden' in combiner_outputs.")

        if hidden.dim() == 2:
            memory = hidden.unsqueeze(1)
        elif hidden.dim() == 3:
            memory = hidden
        else:
            raise ValueError(f"Unexpected encoder output shape: {hidden.shape}")

        if self.memory_projection is not None:
            memory = self.memory_projection(memory)
        return memory

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def _teacher_forced_forward(self, memory: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Parallel teacher-forced forward used during training.

        Args:
            memory: [batch_size, src_len, d_model]
            target: [batch_size, tgt_len] ground-truth token ids.

        Returns:
            logits: [batch_size, tgt_len, vocab_size]
        """
        batch_size, tgt_len = target.shape
        device = target.device

        start = self.start_token.expand(batch_size, 1)
        decoder_input = torch.cat([start, target[:, :-1]], dim=1)

        tgt_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoder.pe[:tgt_len].unsqueeze(0)

        causal_mask = self._causal_mask(tgt_len, device)
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=causal_mask)
        return self.out(out)

    def _autoregressive_forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Autoregressive greedy decoding used at inference.

        Args:
            memory: [batch_size, src_len, d_model]

        Returns:
            logits: [batch_size, max_sequence_length, vocab_size]
        """
        batch_size = memory.size(0)
        device = memory.device
        generated = self.start_token.expand(batch_size, 1).clone()
        all_logits = []

        for _t in range(self.max_sequence_length):
            tgt_len = generated.size(1)
            tgt_emb = self.embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = tgt_emb + self.pos_encoder.pe[:tgt_len].unsqueeze(0)
            causal_mask = self._causal_mask(tgt_len, device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=causal_mask)
            step_logits = self.out(out[:, -1, :])
            all_logits.append(step_logits)
            next_token = step_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return torch.stack(all_logits, dim=1)

    def _beam_forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Beam search decoding at inference when beam_width > 1.

        Args:
            memory: [batch_size, src_len, d_model]

        Returns:
            logits: [batch_size, max_sequence_length, vocab_size] one-hot at selected tokens.
        """
        batch_size = memory.size(0)
        device = memory.device
        start_token = int(self.start_token.item())
        all_logits = []

        for b in range(batch_size):
            mem_b = memory[b : b + 1, :, :].expand(self.beam_width, -1, -1)

            def _step(tokens: torch.Tensor, gen: torch.Tensor):
                new_gen = torch.cat([gen, tokens.unsqueeze(1)], dim=1)
                tgt_len = new_gen.size(1)
                tgt_emb = self.embedding(new_gen) * math.sqrt(self.d_model)
                tgt_emb = tgt_emb + self.pos_encoder.pe[:tgt_len].unsqueeze(0)
                causal_mask = self._causal_mask(tgt_len, device)
                out = self.transformer_decoder(tgt_emb, mem_b, tgt_mask=causal_mask)
                step_logits = self.out(out[:, -1, :])
                return torch.log_softmax(step_logits, dim=-1), new_gen

            def _state_index(gen: torch.Tensor, idx: torch.Tensor):
                return gen[idx]

            init_gen = torch.full((self.beam_width, 1), start_token, dtype=torch.long, device=device)
            seqs, _ = _beam_search_stateful(
                _step,
                init_gen,
                _state_index,
                start_token,
                self.max_sequence_length,
                self.vocab_size,
                self.beam_width,
                self.beam_length_penalty,
                device,
            )

            best_seq = seqs[0]
            item_logits = torch.zeros(self.max_sequence_length, self.vocab_size, device=device)
            for t in range(self.max_sequence_length):
                item_logits[t, best_seq[t]] = 1.0
            all_logits.append(item_logits)

        return torch.stack(all_logits, dim=0)

    def forward(
        self, combiner_outputs: dict[str, torch.Tensor], target: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """Decode combiner outputs into a token sequence.

        Training (target provided): parallel teacher-forced forward pass.
        Inference (target=None): greedy decoding or beam search when beam_width > 1.

        Args:
            combiner_outputs: Dict containing 'hidden' or 'encoder_output'.
            target: [batch_size, max_sequence_length] ground-truth token ids (training only).

        Returns:
            Dict with LOGITS -> [batch_size, max_sequence_length, vocab_size].
        """
        memory = self._get_memory(combiner_outputs)

        if target is not None:
            logits = self._teacher_forced_forward(memory, target)
        elif self.beam_width > 1:
            logits = self._beam_forward(memory)
        else:
            logits = self._autoregressive_forward(memory)

        return {LOGITS: logits}

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    @staticmethod
    def get_schema_cls():
        return TransformerDecoderConfig

    @property
    def input_shape(self):
        return torch.Size([1])

    @property
    def output_shape(self):
        return torch.Size([self.max_sequence_length, self.vocab_size])
