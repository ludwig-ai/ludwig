import pytest
import torch

from ludwig.constants import ENCODER_OUTPUT_STATE, HIDDEN
from ludwig.decoders import sequence_decoder_utils
from ludwig.modules.reduction_modules import SequenceReducer


@pytest.mark.parametrize("num_layers", [1, 2])
def test_get_rnn_init_state_uses_hidden(num_layers):
    batch_size = 16
    sequence_length = 32
    state_size = 64
    combiner_outputs = {}
    combiner_outputs[HIDDEN] = torch.rand([batch_size, sequence_length, state_size])

    # With sequence reduction.
    result = sequence_decoder_utils.get_rnn_init_state(combiner_outputs, SequenceReducer(reduce_mode="sum"), num_layers)
    assert list(result.size()) == [num_layers, batch_size, state_size]

    # Without sequence reduction.
    with pytest.raises(ValueError):
        sequence_decoder_utils.get_rnn_init_state(combiner_outputs, SequenceReducer(reduce_mode="none"), num_layers)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_get_rnn_init_state_prefers_encoder_output_state(num_layers):
    batch_size = 16
    state_size = 64
    combiner_outputs = {}
    combiner_outputs[HIDDEN] = torch.rand([batch_size, state_size])
    combiner_outputs[ENCODER_OUTPUT_STATE] = torch.rand([batch_size, state_size * 2])

    result = sequence_decoder_utils.get_rnn_init_state(combiner_outputs, SequenceReducer(reduce_mode="sum"), num_layers)

    assert list(result.size()) == [num_layers, batch_size, state_size * 2]


@pytest.mark.parametrize("num_layers", [1, 2])
def test_get_lstm_init_state_uses_hidden(num_layers):
    batch_size = 16
    sequence_length = 32
    state_size = 64
    combiner_outputs = {}
    combiner_outputs[HIDDEN] = torch.rand([batch_size, sequence_length, state_size])

    # With sequence reduction.
    decoder_hidden_state, decoder_cell_state = sequence_decoder_utils.get_lstm_init_state(
        combiner_outputs, SequenceReducer(reduce_mode="sum"), num_layers
    )
    assert list(decoder_hidden_state.size()) == [num_layers, batch_size, state_size]
    assert list(decoder_cell_state.size()) == [num_layers, batch_size, state_size]

    # Without sequence reduction.
    with pytest.raises(ValueError):
        sequence_decoder_utils.get_lstm_init_state(combiner_outputs, SequenceReducer(reduce_mode="none"), num_layers)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_get_lstm_init_state_prefers_encoder_output_state(num_layers):
    batch_size = 16
    state_size = 64
    combiner_outputs = {}
    combiner_outputs[HIDDEN] = torch.rand([batch_size, state_size])
    combiner_outputs[ENCODER_OUTPUT_STATE] = torch.rand([batch_size, state_size * 2])

    decoder_hidden_state, decoder_cell_state = sequence_decoder_utils.get_lstm_init_state(
        combiner_outputs, SequenceReducer(reduce_mode="sum"), num_layers
    )

    assert list(decoder_hidden_state.size()) == [num_layers, batch_size, state_size * 2]
    assert list(decoder_cell_state.size()) == [num_layers, batch_size, state_size * 2]
