import pytest
import torch

from ludwig.constants import HIDDEN, LOGITS
from ludwig.decoders.sequence_decoders import (
    LSTMDecoder,
    RNNDecoder,
    SequenceGeneratorDecoder,
    SequenceLSTMDecoder,
    SequenceRNNDecoder,
)
from ludwig.utils.misc_utils import set_random_seed
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919


@pytest.mark.parametrize("cell_type", ["rnn", "gru"])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [20, 1])
def test_rnn_decoder(cell_type, num_layers, batch_size):
    hidden_size = 256
    vocab_size = 50

    input = torch.randint(vocab_size, size=(batch_size,))
    initial_hidden = torch.zeros(num_layers, batch_size, hidden_size)
    rnn_decoder = RNNDecoder(hidden_size, vocab_size, cell_type, num_layers=num_layers)

    output = rnn_decoder(input, initial_hidden)

    assert len(output) == 2
    assert list(output[0].size()) == [batch_size, 1, vocab_size]
    assert list(output[1].size()) == [num_layers, batch_size, hidden_size]


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [20, 1])
def test_lstm_decoder(num_layers, batch_size):
    hidden_size = 256
    vocab_size = 50

    input = torch.randint(vocab_size, size=(batch_size,))
    initial_hidden = torch.zeros(num_layers, batch_size, hidden_size)
    initial_cell_state = torch.zeros(num_layers, batch_size, hidden_size)
    lstm_decoder = LSTMDecoder(hidden_size, vocab_size, num_layers=num_layers)

    output = lstm_decoder(input, initial_hidden, initial_cell_state)

    assert len(output) == 3
    assert list(output[0].size()) == [batch_size, 1, vocab_size]
    assert list(output[1].size()) == [num_layers, batch_size, hidden_size]
    assert list(output[2].size()) == [num_layers, batch_size, hidden_size]


@pytest.mark.parametrize("cell_type", ["rnn", "gru"])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [20, 1])
def test_sequence_rnn_decoder(cell_type, num_layers, batch_size):
    hidden_size = 256
    vocab_size = 50
    max_sequence_length = 10

    # make repeatable
    set_random_seed(RANDOM_SEED)

    combiner_outputs = {HIDDEN: torch.rand([batch_size, hidden_size])}
    sequence_rnn_decoder = SequenceRNNDecoder(
        hidden_size, vocab_size, max_sequence_length, cell_type, num_layers=num_layers
    )

    output = sequence_rnn_decoder(combiner_outputs, target=None)

    assert list(output.size()) == [batch_size, max_sequence_length, vocab_size]

    # check for parameter updating
    target = torch.randn(output.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sequence_rnn_decoder, (combiner_outputs, None), target)
    assert upc == tpc, f"Failed to update parameters.  Parameters not update: {not_updated}"


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [20, 1])
def test_sequence_lstm_decoder(num_layers, batch_size):
    hidden_size = 256
    vocab_size = 50
    max_sequence_length = 10

    # make repeatable
    set_random_seed(RANDOM_SEED)

    combiner_outputs = {HIDDEN: torch.rand([batch_size, hidden_size])}
    sequence_lstm_decoder = SequenceLSTMDecoder(hidden_size, vocab_size, max_sequence_length, num_layers=num_layers)

    output = sequence_lstm_decoder(combiner_outputs, target=None)

    assert list(output.size()) == [batch_size, max_sequence_length, vocab_size]

    # check for parameter updating
    target = torch.randn(output.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(
        sequence_lstm_decoder, (combiner_outputs, None), target
    )
    assert upc == tpc, f"Failed to update parameters.  Parameters not update: {not_updated}"


@pytest.mark.parametrize("cell_type", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [20, 1])
def test_sequence_generator_decoder(cell_type, num_layers, batch_size):
    hidden_size = 256
    vocab_size = 50
    max_sequence_length = 10

    # make repeatable
    set_random_seed(RANDOM_SEED)

    combiner_outputs = {HIDDEN: torch.rand([batch_size, hidden_size])}
    sequence_rnn_decoder = SequenceGeneratorDecoder(
        input_size=hidden_size,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        cell_type=cell_type,
        num_layers=num_layers,
    )

    output = sequence_rnn_decoder(combiner_outputs, target=None)

    assert list(output[LOGITS].size()) == [batch_size, max_sequence_length, vocab_size]

    # check for parameter updating
    target = torch.randn(output[LOGITS].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sequence_rnn_decoder, (combiner_outputs, None), target)
    assert upc == tpc, f"Failed to update parameters.  Parameters not update: {not_updated}"
