"""Utility functions related to sequence decoders."""

from typing import Dict, Tuple

import torch

from ludwig.constants import ENCODER_OUTPUT_STATE, HIDDEN
from ludwig.modules.reduction_modules import SequenceReducer


def repeat_2D_tensor(tensor, k):
    """Repeats a 2D-tensor k times over the first dimension.

    For example:
    Input: Tensor of [batch_size, state_size], k=2
    Output: Tensor of [k, batch_size, state_size]
    """
    if len(tensor.size()) > 2:
        raise ValueError("Cannot repeat a non-2D tensor with this method.")
    return tensor.repeat(k, 1, 1)


def get_rnn_init_state(
    combiner_outputs: Dict[str, torch.Tensor], sequence_reducer: SequenceReducer, num_layers: int
) -> torch.Tensor:
    """Computes the hidden state that the RNN decoder should start with.

    Args:
        combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.
        sequence_reducer: SequenceReducer to reduce rank-3 to rank-2.
        num_layers: Number of layers the decoder uses.

    Returns:
        Tensor of [num_layers, batch_size, hidden_size].
    """
    if ENCODER_OUTPUT_STATE not in combiner_outputs:
        # Use the combiner's hidden state.
        encoder_output_state = combiner_outputs[HIDDEN]
    else:
        # Use the encoder's output state.
        encoder_output_state = combiner_outputs[ENCODER_OUTPUT_STATE]
        if isinstance(encoder_output_state, tuple):
            if len(encoder_output_state) == 2:
                # LSTM encoder. Use the hidden state and ignore the cell state.
                encoder_output_state = encoder_output_state[0]
            elif len(encoder_output_state) == 4:
                # Bi-directional LSTM encoder. Use the average of hidden states and ignore cell state.
                encoder_output_state = torch.mean([encoder_output_state[0], encoder_output_state[2]])
            else:
                raise ValueError(
                    f"Invalid sequence decoder inputs with keys: {combiner_outputs.keys()} with extracted encoder "
                    + f"state: {encoder_output_state.size()} that was invalid. Please double check the compatibility "
                    + "of your encoder and decoder."
                )

    if len(encoder_output_state.size()) > 3:
        raise ValueError("Init state for RNN decoders only works for 1d or 2d tensors (encoder_output).")

    if len(encoder_output_state.size()) == 3:
        # Reduce to [batch_size, hidden_size].
        encoder_output_state = sequence_reducer(encoder_output_state)

    return repeat_2D_tensor(encoder_output_state, num_layers)


def get_lstm_init_state(
    combiner_outputs: Dict[str, torch.Tensor], sequence_reducer: SequenceReducer, num_layers: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the states that the LSTM decoder should start with.

    Args:
        combiner_outputs: Dictionary of tensors from the outputs of the combiner and other output features.
        sequence_reducer: SequenceReducer to reduce rank-3 to rank-2.
        num_layers: Number of layers the decoder uses.

    Returns:
        Tuple of 2 tensors (decoder hidden state, decoder cell state), each [num_layers, batch_size, hidden_size].
    """
    if "encoder_output_state" not in combiner_outputs:
        # Use the combiner's hidden state.
        decoder_hidden_state = combiner_outputs[HIDDEN]
        decoder_cell_state = torch.clone(decoder_hidden_state)
    else:
        # Use the encoder's output state.
        encoder_output_state = combiner_outputs[ENCODER_OUTPUT_STATE]
        if not isinstance(encoder_output_state, tuple):
            decoder_hidden_state = encoder_output_state
            decoder_cell_state = decoder_hidden_state
        else:
            if len(encoder_output_state) == 2:
                # The encoder was probably an LSTM.
                decoder_hidden_state, decoder_cell_state = encoder_output_state
            elif len(encoder_output_state) == 4:
                # The encoder was probably a bi-LSTM.
                # Use the average of the encoder's hidden states for hidden state.
                # Use the average of the encoder's cell states for cell state.
                decoder_hidden_state = torch.mean([encoder_output_state[0], encoder_output_state[2]])
                decoder_cell_state = torch.mean([encoder_output_state[1], encoder_output_state[3]])
            else:
                raise ValueError(
                    f"Invalid sequence decoder inputs with keys: {combiner_outputs.keys()} with extracted encoder "
                    + f"state: {encoder_output_state} that was invalid. Please double check the compatibility of your "
                    + "encoder and decoder."
                )

    # Check rank and reduce if necessary.
    if len(decoder_hidden_state.size()) > 3 or len(decoder_cell_state.size()) > 3:
        raise ValueError(
            f"Invalid sequence decoder inputs with keys: {combiner_outputs.keys()} with extracted encoder "
            + f"state: {decoder_hidden_state.size()} that was invalid. Please double check the compatibility "
            + "of your encoder and decoder."
        )
    if len(decoder_hidden_state.size()) == 3:
        decoder_hidden_state = sequence_reducer(decoder_hidden_state)
    if len(decoder_cell_state.size()) == 3:
        decoder_cell_state = sequence_reducer(decoder_cell_state)

    # Repeat over the number of layers.
    return repeat_2D_tensor(decoder_hidden_state, num_layers), repeat_2D_tensor(decoder_cell_state, num_layers)
